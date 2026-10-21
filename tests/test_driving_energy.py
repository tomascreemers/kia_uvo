"""Tests for the statistics-safe driving energy and distance meters."""

import asyncio
import copy
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.components.sensor import SensorStateClass
from homeassistant.util import dt as dt_util
from hyundai_kia_connect_api import Vehicle
from hyundai_kia_connect_api.Vehicle import DailyDrivingStats

from custom_components.kia_uvo import sensor as sensor_platform
from custom_components.kia_uvo.const import CONF_LIFETIME_AVERAGE_CONSUMPTION
from custom_components.kia_uvo.driving_energy import (
    DailyTotal,
    DrivingMeter,
    DrivingMeterTracker,
    backfill_statistics,
    daily_distance_rows,
)
from tests.conftest import make_coordinator, make_hass

TZ = ZoneInfo("Europe/Brussels")
NOW = datetime(2026, 10, 8, 21, 7, tzinfo=TZ)


def rows(**days: float) -> dict[date, float]:
    """Rows keyed like ``d20261006=13089``."""
    return {datetime.strptime(k[1:], "%Y%m%d").date(): v for k, v in days.items()}


def daily_stats(energy_rows: dict[date, float]) -> list[DailyDrivingStats]:
    """The library's daily rows, at 160 Wh/km."""
    return [
        DailyDrivingStats(
            date=datetime(day.year, day.month, day.day),
            total_consumed=wh,
            distance=round(wh / 160),
        )
        for day, wh in energy_rows.items()
    ]


def ev(energy_rows: dict[date, float]) -> Vehicle:
    vehicle = Vehicle(id="v1", name="test", model="EV6")
    vehicle.odometer = (50828.1, "km")
    vehicle.total_power_consumed = 959700
    vehicle.daily_stats = daily_stats(energy_rows)
    return vehicle


class FakeStore:
    """In-memory stand-in for ``homeassistant.helpers.storage.Store``."""

    def __init__(self) -> None:
        self.data: dict[str, Any] | None = None
        self.saves = 0

    async def async_load(self) -> dict[str, Any] | None:
        return self.data

    async def async_save(self, data: dict[str, Any]) -> None:
        self.data = data
        self.saves += 1


class FakeRecorder:
    """Records ``async_adjust_statistics`` calls.

    A test can set ``commit`` to a pending future to hold back the commit.
    """

    def __init__(self) -> None:
        self.adjusted: list[tuple[str, datetime, float, str]] = []
        self.commit: asyncio.Future[None] | None = None

    def async_get_commit_future(self) -> asyncio.Future[None] | None:
        return self.commit

    def async_adjust_statistics(
        self, statistic_id: str, start: datetime, adjustment: float, unit: str
    ) -> None:
        self.adjusted.append((statistic_id, start, adjustment, unit))


def started_meter() -> DrivingMeter:
    meter = DrivingMeter()
    meter.update(
        rows(d20261005=26762, d20261006=13089),
        rows(d20261005=161, d20261006=78),
        NOW,
        odometer_km=50828.1,
        api_wh_per_km=158.0,
    )
    return meter


# The meter


def test_first_rows_only_set_the_baseline() -> None:
    meter = started_meter()
    assert meter.energy.initialized
    assert meter.energy.total == 0
    assert meter.distance.total == 0
    assert meter.odometer_at_start_km == 50828.1


def test_late_past_day_is_kept_apart_from_today() -> None:
    meter = started_meter()
    # Recorded 2026-10-08: the 6 Oct return trip arrived 39 h late,
    # together with the first drive of 8 Oct.
    meter.update(rows(d20261005=26762, d20261006=25516, d20261008=13238), {}, NOW)
    assert meter.energy.total == 13238
    assert meter.energy.late == rows(d20261006=12427)
    meter.update(rows(d20261005=26762, d20261006=25516, d20261008=19465), {}, NOW)
    assert meter.energy.total == 19465
    assert meter.energy.measured == 31892


def test_trip_uploaded_two_days_late() -> None:
    # Recorded 2026-10-10: Thursday's drive home was only uploaded with
    # Saturday's first trip, as 42 km more on the 8 Oct row.
    meter = DrivingMeter()
    thursday = datetime(2026, 10, 8, 21, 7, tzinfo=TZ)
    meter.update(
        rows(d20261006=25516, d20261008=19465),
        rows(d20261006=156, d20261008=116),
        thursday,
    )
    saturday = datetime(2026, 10, 10, 9, tzinfo=TZ)
    meter.update(
        rows(d20261006=25516, d20261008=26620, d20261010=221),
        rows(d20261006=156, d20261008=158, d20261010=2),
        saturday,
    )
    assert meter.energy.total == 221
    assert meter.energy.late == rows(d20261008=7155)
    assert meter.distance.total == 2
    assert meter.distance.late == rows(d20261008=42)
    assert meter.energy.late_hour(date(2026, 10, 8), TZ) == datetime(
        2026, 10, 8, 23, tzinfo=TZ
    )


def test_late_change_before_the_start_goes_to_the_backfilled_hour() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27000, d20261006=13089), {}, NOW)
    assert meter.energy.late == rows(d20261005=238)
    assert meter.energy.late_hour(date(2026, 10, 5), TZ) == datetime(
        2026, 10, 5, 12, tzinfo=TZ
    )


def test_new_empty_row_is_not_late() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=26762, d20261006=13089, d20261007=0), {}, NOW)
    assert meter.energy.late == {}
    assert date(2026, 10, 7) in meter.energy.rows


def test_mark_backdated_moves_late_into_backdated() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27000, d20261006=14089), {}, NOW)
    applied = dict(meter.energy.late)
    # More arrives for the same day before the first part is marked.
    meter.update(rows(d20261005=27100, d20261006=14089), {}, NOW)
    meter.energy.mark_backdated(applied)
    assert meter.energy.late == rows(d20261005=100)
    assert meter.energy.backdated == 1238
    assert meter.energy.measured == 1338
    meter.energy.mark_backdated(rows(d20261005=100))
    assert meter.energy.late == {}


def test_empty_response_changes_nothing() -> None:
    meter = started_meter()
    before = copy.deepcopy(meter)
    meter.update({}, {}, NOW, window_wh=970000)
    assert meter == before
    meter.update(rows(d20261005=26762, d20261006=13089), {}, NOW)
    assert meter.energy.measured == 0


def test_row_missing_from_one_response_is_not_counted_again() -> None:
    meter = started_meter()
    meter.update(rows(d20261006=13089), {}, NOW)
    meter.update(rows(d20261005=26762, d20261006=13089), {}, NOW)
    assert meter.energy.measured == 0


def test_downward_correction_is_subtracted() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=26000, d20261006=13089, d20261008=500), {}, NOW)
    meter.update(rows(d20261005=26000, d20261006=13089, d20261008=400), {}, NOW)
    assert meter.energy.late == rows(d20261005=-762)
    assert meter.energy.total == 400


def test_rows_are_forgotten_by_age_and_old_rows_ignored() -> None:
    meter = started_meter()
    later = datetime(2026, 11, 12, 9, tzinfo=TZ)
    meter.update(rows(d20261110=5000), {}, later)
    assert meter.energy.rows == rows(d20261110=5000)
    assert meter.energy.late == rows(d20261110=5000)
    # A row older than the retention window must never be counted.
    meter.update(rows(d20261005=26762, d20261110=5000), {}, later)
    assert meter.energy.measured == 5000


def test_distance_starts_on_its_own() -> None:
    meter = started_meter()
    meter.distance = DailyTotal()
    meter.update({}, rows(d20261006=78), NOW)
    assert meter.distance.initialized
    assert meter.distance.total == 0


def test_distance_rows_in_miles_are_converted() -> None:
    vehicle = ev({})
    vehicle.daily_stats = [
        DailyDrivingStats(
            date=datetime(2026, 10, 6),
            total_consumed=100,
            distance=10,
            distance_unit="mi",
        )
    ]
    assert daily_distance_rows(vehicle) == {date(2026, 10, 6): pytest.approx(16.09344)}


def test_lifetime_estimate() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27762, d20261006=13089, d20261008=1000), {}, NOW)
    measured = 1000 + 1000
    assert meter.lifetime_estimate_wh(185.0) == pytest.approx(50828.1 * 185 + measured)
    assert meter.lifetime_estimate_wh(None) == pytest.approx(50828.1 * 158 + measured)
    assert DrivingMeter().lifetime_estimate_wh(185.0) is None


def test_round_trip() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27762, d20261006=13089, d20261008=1000), {}, NOW)
    meter.energy.mark_backdated(rows(d20261005=400))
    assert DrivingMeter.from_dict(meter.as_dict()) == meter


def test_stored_format() -> None:
    """The stored format must not change: meters already run on it."""
    meter = DrivingMeter()
    meter.update(rows(d20261006=13089), {}, NOW, window_wh=962025)
    meter.update(rows(d20261005=1000, d20261006=13089), {}, NOW, window_wh=963025)
    assert meter.as_dict() == {
        "energy": {
            "total": 0.0,
            "rows": {"2026-10-05": 1000.0, "2026-10-06": 13089.0},
            "started_at": "2026-10-08T21:07:00+02:00",
            "baseline_rows": {"2026-10-06": 13089.0},
            "backfilled": False,
            "late": {"2026-10-05": 1000.0},
            "backdated": 0.0,
        },
        "distance": {
            "total": 0.0,
            "rows": {},
            "started_at": None,
            "baseline_rows": {},
            "backfilled": False,
            "late": {},
            "backdated": 0.0,
        },
        "odometer_at_start_km": None,
        "api_wh_per_km_at_start": None,
        "window_wh": 963025.0,
        "window_carry_wh": 0.0,
        "missed_wh": 0.0,
    }


# Driving the meters cannot count, seen in the 90-day total


def window_meter() -> DrivingMeter:
    meter = DrivingMeter()
    meter.update(rows(d20261006=25516, d20261008=19465), {}, NOW, window_wh=962025)
    return meter


def test_window_total_matching_the_daily_rows_misses_nothing() -> None:
    # Recorded 2026-10-10: both grew by 7155 + 221 Wh.
    meter = window_meter()
    saturday = datetime(2026, 10, 10, 9, tzinfo=TZ)
    meter.update(
        rows(d20261006=25516, d20261008=26620, d20261010=221),
        {},
        saturday,
        window_wh=969401,
    )
    assert meter.missed_wh == 0
    assert meter.window_wh == 969401


def test_window_total_growing_beyond_the_daily_rows_is_missed() -> None:
    # A trip from 40 days ago arrives with today's drive: only the 90-day
    # total shows it.
    meter = window_meter()
    meter.update(
        rows(d20261006=25516, d20261008=19465), {}, NOW, window_wh=962025 + 5000
    )
    assert meter.missed_wh == 5000


def test_trip_between_the_two_requests_is_not_missed() -> None:
    meter = window_meter()
    # 90-day total fetched just before the trip arrived, daily rows after.
    meter.update(rows(d20261006=25516, d20261008=20465), {}, NOW, window_wh=962025)
    meter.update(rows(d20261006=25516, d20261008=20465), {}, NOW, window_wh=963025)
    assert meter.missed_wh == 0
    assert meter.window_carry_wh == 0


def test_day_dropping_out_is_not_missed_and_not_carried_on() -> None:
    meter = window_meter()
    # The window's oldest day, with 11 kWh of driving, drops out.
    meter.update(rows(d20261006=25516, d20261008=19465), {}, NOW, window_wh=951025)
    meter.update(rows(d20261006=25516, d20261008=19465), {}, NOW, window_wh=951025)
    meter.update(rows(d20261006=25516, d20261008=19465), {}, NOW, window_wh=954025)
    assert meter.missed_wh == 3000


def test_late_trip_smaller_than_the_day_dropping_out_goes_unnoticed() -> None:
    meter = window_meter()
    # 5 kWh reported 40 days late, while 11 kWh drops out of the window.
    meter.update(rows(d20261006=25516, d20261008=19465), {}, NOW, window_wh=956025)
    meter.update(rows(d20261006=25516, d20261008=19465), {}, NOW, window_wh=956025)
    assert meter.missed_wh == 0


def test_window_total_needs_daily_rows() -> None:
    meter = window_meter()
    meter.update({}, {}, NOW, window_wh=970000)
    assert meter.window_wh == 962025
    meter.update(rows(d20261006=25516, d20261008=27440), {}, NOW, window_wh=970000)
    assert meter.missed_wh == 0


# The backfill


def test_backfill_starts_at_zero_before_first_state() -> None:
    meter = DrivingMeter()
    meter.update(rows(d20261005=26762, d20261006=25516, d20261008=19465), {}, NOW)
    backfill = backfill_statistics(meter.energy, TZ)
    assert backfill is not None
    assert backfill.rows == [
        (datetime(2026, 10, 5, 12, tzinfo=TZ), 26762),
        (datetime(2026, 10, 6, 12, tzinfo=TZ), 52278),
        (datetime(2026, 10, 8, 12, tzinfo=TZ), 71743),
    ]
    assert backfill.total == 71743
    assert backfill.meter_start == datetime(2026, 10, 8, 21, tzinfo=TZ)


def test_backfill_moves_start_day_before_a_morning_install() -> None:
    meter = DrivingMeter()
    morning = datetime(2026, 10, 8, 9, 30, tzinfo=TZ)
    meter.update(rows(d20261008=500), {}, morning)
    backfill = backfill_statistics(meter.energy, TZ)
    assert backfill is not None
    # The last full hour before the meter's first state.
    assert backfill.rows == [(datetime(2026, 10, 8, 8, tzinfo=TZ), 500)]
    assert backfill.meter_start == datetime(2026, 10, 8, 9, tzinfo=TZ)


def test_backfill_merges_days_that_share_an_hour() -> None:
    meter = DrivingMeter()
    morning = datetime(2026, 10, 8, 9, 30, tzinfo=TZ)
    # A row dated after the local date, from a car in another time zone.
    meter.update(rows(d20261007=300, d20261008=500, d20261009=200), {}, morning)
    backfill = backfill_statistics(meter.energy, TZ)
    assert backfill is not None
    assert backfill.rows == [
        (datetime(2026, 10, 7, 12, tzinfo=TZ), 300),
        (datetime(2026, 10, 8, 8, tzinfo=TZ), 1000),
    ]
    assert backfill.total == 1000


def test_backfill_needs_a_started_meter() -> None:
    assert backfill_statistics(DailyTotal(), TZ) is None


# The tracker


async def test_tracker_saves_before_publishing() -> None:
    store = FakeStore()
    vehicle = ev(rows(d20261006=13089))
    tracker = await DrivingMeterTracker.async_load(store, vehicle, lambda: NOW)
    await tracker.async_process()

    published: list[float] = []

    def listener() -> None:
        assert store.data is not None
        assert store.data["energy"]["total"] == tracker.meter.energy.total
        published.append(tracker.meter.energy.total)

    tracker.add_listener(listener)
    vehicle.daily_stats = daily_stats(rows(d20261006=13089, d20261008=13238))
    vehicle.total_power_consumed += 13238
    await tracker.async_process()
    saves = store.saves
    await tracker.async_process()  # nothing changed
    assert published == [13238]
    assert store.saves == saves


async def test_tracker_resumes_from_store() -> None:
    store = FakeStore()
    tracker = await DrivingMeterTracker.async_load(
        store, ev(rows(d20261006=13089)), lambda: NOW
    )
    await tracker.async_process()

    vehicle = ev(rows(d20261006=13089, d20261008=13238))
    resumed = await DrivingMeterTracker.async_load(store, vehicle, lambda: NOW)
    await resumed.async_process()
    assert resumed.meter.energy.total == 13238
    assert resumed.meter.distance.total == 83


async def test_tracker_ignores_a_vehicle_without_daily_rows() -> None:
    store = FakeStore()
    vehicle = ev(rows(d20261006=13089))
    tracker = await DrivingMeterTracker.async_load(store, vehicle, lambda: NOW)
    await tracker.async_process()
    saves = store.saves
    vehicle.daily_stats = None
    await tracker.async_process()
    assert store.saves == saves


async def test_tracker_warns_about_missed_driving(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FakeStore()
    vehicle = ev(rows(d20261006=13089))
    tracker = await DrivingMeterTracker.async_load(store, vehicle, lambda: NOW)
    await tracker.async_process()
    vehicle.total_power_consumed += 4000
    vehicle.daily_stats = daily_stats(rows(d20261006=13089, d20261008=1000))
    await tracker.async_process()
    assert store.data is not None and store.data["missed_wh"] == 3000
    assert "3000 Wh of driving was reported for days older" in caplog.text


async def test_tracker_marks_backfilled_and_backdated() -> None:
    store = FakeStore()
    vehicle = ev(rows(d20261006=13089))
    tracker = await DrivingMeterTracker.async_load(store, vehicle, lambda: NOW)
    await tracker.async_process()
    await tracker.async_mark_backfilled("distance")
    assert store.data is not None
    assert store.data["distance"]["backfilled"]
    assert not store.data["energy"]["backfilled"]
    assert store.data["energy"]["baseline_rows"] == {"2026-10-06": 13089}

    vehicle.daily_stats = daily_stats(rows(d20261006=14089))
    await tracker.async_process()
    assert store.data["energy"]["late"] == {"2026-10-06": 1000}
    await tracker.async_mark_backdated("energy", rows(d20261006=1000))
    assert store.data["energy"]["late"] == {}
    assert store.data["energy"]["backdated"] == 1000


# The sensors


async def _set_up_sensors(
    monkeypatch: pytest.MonkeyPatch,
    vehicle: Vehicle,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the sensors set up for ``vehicle``, by unique id less its prefix."""
    monkeypatch.setattr(sensor_platform, "Store", lambda *args: FakeStore())
    hass, config_entry = make_hass(make_coordinator(vehicle))
    config_entry.options = options or {}
    created: list[Any] = []
    await sensor_platform.async_setup_entry(hass, config_entry, created.extend)
    return {entity.unique_id.removeprefix("kia_uvo_v1_"): entity for entity in created}


async def test_sensor_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    sensors = await _set_up_sensors(monkeypatch, ev(rows(d20261006=13089)))
    energy = sensors["driving_energy"]
    distance = sensors["driving_distance"]
    assert energy.state_class == SensorStateClass.TOTAL
    assert energy.native_value == 0
    assert energy.unique_id == "kia_uvo_v1_driving_energy"
    assert energy.native_unit_of_measurement == "Wh"
    assert distance.state_class == SensorStateClass.TOTAL
    assert distance.native_value == 0
    assert distance.unique_id == "kia_uvo_v1_driving_distance"
    assert distance.native_unit_of_measurement == "km"


async def test_lifetime_estimate_sensor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dt_util, "now", lambda: NOW)
    sensors = await _set_up_sensors(
        monkeypatch,
        ev(rows(d20261006=13089)),
        {CONF_LIFETIME_AVERAGE_CONSUMPTION: 18.5},
    )
    estimate = sensors["lifetime_driving_energy_estimate"]
    assert estimate.state_class is None
    assert estimate.native_value == pytest.approx(50828.1 * 185)
    assert estimate.extra_state_attributes == {
        "measured_since": "2026-10-08",
        "odometer_at_start_km": 50828.1,
        "average_consumption_kwh_per_100km": 18.5,
        "average_source": "configured",
    }


async def _started_sensors(
    monkeypatch: pytest.MonkeyPatch, vehicle: Vehicle, compiled: bool = True
) -> tuple[dict[str, Any], list[tuple[Any, list[Any]]], FakeRecorder]:
    """Set up the sensors as added to a hass with a recorder."""
    imported: list[tuple[Any, list[Any]]] = []
    monkeypatch.setattr(
        sensor_platform,
        "async_import_statistics",
        lambda hass, metadata, stats: imported.append((metadata, list(stats))),
    )
    fake_recorder = FakeRecorder()
    monkeypatch.setattr(sensor_platform, "get_instance", lambda hass: fake_recorder)
    monkeypatch.setattr(dt_util, "now", lambda: NOW)
    monkeypatch.setattr(dt_util, "get_default_time_zone", lambda: TZ)
    sensors = {
        key: sensor
        for key, sensor in (await _set_up_sensors(monkeypatch, vehicle)).items()
        if isinstance(sensor, sensor_platform.DrivingMeterSensor)
    }

    async def has_short_term_statistics(self: Any) -> bool:
        return compiled

    monkeypatch.setattr(
        sensor_platform.DrivingMeterSensor,
        "_async_has_short_term_statistics",
        has_short_term_statistics,
    )
    monkeypatch.setattr(
        sensor_platform.DrivingMeterSensor,
        "unit_of_measurement",
        property(lambda self: "kWh" if self.device_class == "energy" else "km"),
    )
    for key, sensor in sensors.items():
        sensor.hass = MagicMock()
        sensor.hass.config.components = {"recorder"}
        sensor.entity_id = f"sensor.ev6_{key}"
        sensor.async_write_ha_state = MagicMock()
    return sensors, imported, fake_recorder


async def test_sensor_imports_backfill_once(monkeypatch: pytest.MonkeyPatch) -> None:
    vehicle = ev(rows(d20261005=26762, d20261006=13089))
    sensors, imported, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    energy = sensors["driving_energy"]

    await energy._async_update_statistics()
    await energy._async_update_statistics()

    assert len(imported) == 1
    metadata, stats = imported[0]
    assert metadata["statistic_id"] == "sensor.ev6_driving_energy"
    assert metadata["source"] == "recorder"
    assert metadata["unit_class"] == "energy"
    assert metadata["unit_of_measurement"] == "kWh"
    assert [s["sum"] for s in stats] == [26.762, pytest.approx(39.851)]
    # The sums compiled since the start continue from the imported total.
    assert fake_recorder.adjusted == [
        (
            "sensor.ev6_driving_energy",
            datetime(2026, 10, 8, 21, tzinfo=TZ),
            39851,
            "Wh",
        )
    ]

    await sensors["driving_distance"]._async_update_statistics()
    metadata, stats = imported[1]
    assert metadata["unit_class"] == "distance"
    assert [s["sum"] for s in stats] == [167, 249]


async def test_sensor_waits_for_short_term_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = ev(rows(d20261005=26762, d20261006=13089))
    sensors, imported, fake_recorder = await _started_sensors(
        monkeypatch, vehicle, compiled=False
    )
    energy = sensors["driving_energy"]
    await energy._async_update_statistics()
    assert imported == []
    assert fake_recorder.adjusted == []
    assert not energy._tracker.meter.energy.backfilled


async def test_sensor_books_late_change_on_its_day_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = ev(rows(d20261006=13089))
    sensors, _, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    energy = sensors["driving_energy"]
    tracker = energy._tracker
    await energy._async_update_statistics()
    fake_recorder.adjusted.clear()

    vehicle.daily_stats = daily_stats(rows(d20261006=14689, d20261008=500))
    await tracker.async_process()
    assert energy.native_value == 500
    await energy._async_update_statistics()
    await energy._async_update_statistics()

    assert fake_recorder.adjusted == [
        (
            "sensor.ev6_driving_energy",
            datetime(2026, 10, 6, 12, tzinfo=TZ),
            1600,
            "Wh",
        )
    ]
    assert tracker.meter.energy.late == {}
    assert energy.native_value == 500
    assert energy.extra_state_attributes == {
        "measured_since": "2026-10-08",
        "booked_on_earlier_days_wh": 1600,
    }
    assert tracker.meter.energy.measured == 2100
    assert tracker.meter.lifetime_estimate_wh(158.0) == pytest.approx(
        50828.1 * 158 + 2100
    )


async def test_sensor_marks_changes_done_only_once_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash before the commit must leave the change to be made again."""
    vehicle = ev(rows(d20261005=26762, d20261006=13089))
    sensors, imported, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    energy = sensors["driving_energy"]
    tracker = energy._tracker

    fake_recorder.commit = asyncio.get_running_loop().create_future()
    update = asyncio.create_task(energy._async_update_statistics())
    await asyncio.sleep(0)
    assert len(imported) == 1
    assert not tracker.meter.energy.backfilled
    fake_recorder.commit.set_result(None)
    await update
    assert tracker.meter.energy.backfilled

    vehicle.daily_stats = daily_stats(rows(d20261005=26762, d20261006=14689))
    await tracker.async_process()
    fake_recorder.commit = asyncio.get_running_loop().create_future()
    update = asyncio.create_task(energy._async_update_statistics())
    await asyncio.sleep(0)
    assert fake_recorder.adjusted[-1][2] == 1600
    assert tracker.meter.energy.late == rows(d20261006=1600)
    fake_recorder.commit.set_result(None)
    await update
    assert tracker.meter.energy.late == {}
    assert tracker.meter.energy.backdated == 1600


async def test_sensor_without_recorder_keeps_late_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = ev(rows(d20261006=13089))
    sensors, _, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    energy = sensors["driving_energy"]
    energy.hass.config.components = set()
    vehicle.daily_stats = daily_stats(rows(d20261006=14689))
    await energy._tracker.async_process()
    await energy._async_update_statistics()
    assert fake_recorder.adjusted == []
    assert energy._tracker.meter.energy.late == rows(d20261006=1600)
