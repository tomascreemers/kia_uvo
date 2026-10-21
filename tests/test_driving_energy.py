"""Tests for the statistics-safe driving energy meter."""

import asyncio
from datetime import date, datetime
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest
from homeassistant.components.sensor import SensorStateClass
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


def test_first_rows_only_set_the_baseline() -> None:
    meter = started_meter()
    assert meter.initialized
    assert meter.energy.total == 0
    assert meter.distance.total == 0
    assert meter.odometer_at_start_km == 50828.1


def test_late_past_day_is_kept_apart_from_today() -> None:
    meter = started_meter()
    # Recorded 2026-10-08: the 6 Oct return trip arrived 39 h late,
    # together with the first drive of 8 Oct.
    meter.update(rows(d20261005=26762, d20261006=25516, d20261008=13238), None, NOW)
    assert meter.energy.total == 13238
    assert meter.energy.late == {"2026-10-06": 12427}
    meter.update(rows(d20261005=26762, d20261006=25516, d20261008=19465), None, NOW)
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
    assert meter.energy.late == {"2026-10-08": 7155}
    assert meter.distance.total == 2
    assert meter.distance.late == {"2026-10-08": 42}
    assert meter.energy.late_hour("2026-10-08", TZ) == datetime(
        2026, 10, 8, 23, tzinfo=TZ
    )


def test_late_change_before_the_start_goes_to_the_backfilled_hour() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27000, d20261006=13089), None, NOW)
    assert meter.energy.late == {"2026-10-05": 238}
    assert meter.energy.late_hour("2026-10-05", TZ) == datetime(
        2026, 10, 5, 12, tzinfo=TZ
    )


def test_mark_backdated_moves_late_into_backdated() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27000, d20261006=14089), None, NOW)
    applied = dict(meter.energy.late)
    # More arrives for the same day before the first part is marked.
    meter.update(rows(d20261005=27100, d20261006=14089), None, NOW)
    meter.energy.mark_backdated(applied)
    assert meter.energy.late == {"2026-10-05": 100}
    assert meter.energy.backdated == 1238
    assert meter.energy.measured == 1338
    meter.energy.mark_backdated({"2026-10-05": 100})
    assert meter.energy.late == {}


@pytest.mark.parametrize("empty", [None, {}])
def test_empty_response_changes_nothing(empty: dict[date, float] | None) -> None:
    meter = started_meter()
    assert not meter.update(empty, empty, NOW)
    meter.update(rows(d20261005=26762, d20261006=13089), None, NOW)
    assert meter.energy.measured == 0


def test_row_missing_from_one_response_is_not_counted_again() -> None:
    meter = started_meter()
    meter.update(rows(d20261006=13089), None, NOW)
    meter.update(rows(d20261005=26762, d20261006=13089), None, NOW)
    assert meter.energy.measured == 0


def test_downward_correction_is_subtracted() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=26000, d20261006=13089, d20261008=500), None, NOW)
    meter.update(rows(d20261005=26000, d20261006=13089, d20261008=400), None, NOW)
    assert meter.energy.late == {"2026-10-05": -762}
    assert meter.energy.total == 400


def test_rows_are_forgotten_by_age_and_old_rows_ignored() -> None:
    meter = started_meter()
    later = datetime(2026, 11, 12, 9, tzinfo=TZ)
    meter.update(rows(d20261110=5000), None, later)
    assert set(meter.energy.rows) == {"2026-11-10"}
    assert meter.energy.late == {"2026-11-10": 5000}
    # A row older than the retention window must never be counted.
    meter.update(rows(d20261005=26762, d20261110=5000), None, later)
    assert meter.energy.measured == 5000


def test_distance_starts_on_its_own() -> None:
    meter = started_meter()
    meter.distance = DailyTotal()
    meter.update(None, rows(d20261006=78), NOW)
    assert meter.distance.initialized
    assert meter.distance.total == 0


def test_lifetime_estimate() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27762, d20261006=13089, d20261008=1000), None, NOW)
    measured = 1000 + 1000
    assert meter.lifetime_estimate_wh(185.0) == pytest.approx(50828.1 * 185 + measured)
    assert meter.lifetime_estimate_wh(None) == pytest.approx(50828.1 * 158 + measured)
    assert DrivingMeter().lifetime_estimate_wh(185.0) is None


def test_round_trip() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27762, d20261006=13089, d20261008=1000), None, NOW)
    meter.energy.mark_backdated({"2026-10-05": 400})
    assert DrivingMeter.from_dict(meter.as_dict()) == meter


def months_meter() -> DrivingMeter:
    meter = DrivingMeter()
    meter.update(rows(d20261006=25516, d20261008=19465), None, NOW, months_wh=962025)
    return meter


def test_months_total_matching_the_daily_rows_misses_nothing() -> None:
    # Recorded 2026-10-10: both grew by 7155 + 221 Wh.
    meter = months_meter()
    saturday = datetime(2026, 10, 10, 9, tzinfo=TZ)
    meter.update(
        rows(d20261006=25516, d20261008=26620, d20261010=221),
        None,
        saturday,
        months_wh=969401,
    )
    assert meter.missed_wh == 0
    assert meter.months_wh == 969401


def test_months_total_growing_beyond_the_daily_rows_is_missed() -> None:
    # A trip from 40 days ago arrives with today's drive: only the monthly
    # total shows it.
    meter = months_meter()
    meter.update(
        rows(d20261006=25516, d20261008=19465), None, NOW, months_wh=962025 + 5000
    )
    assert meter.missed_wh == 5000


def test_trip_between_the_two_requests_is_not_missed() -> None:
    meter = months_meter()
    # Monthly total fetched just before the trip arrived, daily rows after.
    meter.update(rows(d20261006=25516, d20261008=20465), None, NOW, months_wh=962025)
    meter.update(rows(d20261006=25516, d20261008=20465), None, NOW, months_wh=963025)
    assert meter.missed_wh == 0
    assert meter.months_carry_wh == 0


def test_month_dropping_out_is_not_missed_and_not_carried_on() -> None:
    meter = months_meter()
    november = datetime(2026, 11, 1, 9, tzinfo=TZ)
    # July drops out of the monthly total.
    meter.update(rows(d20261008=19465), None, november, months_wh=962025 - 234990)
    meter.update(rows(d20261008=19465), None, november, months_wh=962025 - 234990)
    meter.update(
        rows(d20261008=19465), None, november, months_wh=962025 - 234990 + 3000
    )
    assert meter.missed_wh == 3000


def test_months_total_needs_daily_rows() -> None:
    meter = months_meter()
    assert not meter.update(None, None, NOW, months_wh=970000)
    meter.update(rows(d20261006=25516, d20261008=27440), None, NOW, months_wh=970000)
    assert meter.missed_wh == 0


async def test_tracker_warns_about_missed_driving(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FakeStore()
    vehicle = ev(rows(d20261006=13089))
    tracker = DrivingMeterTracker(store, vehicle, lambda: NOW)
    await tracker.async_load()
    await tracker.async_process()
    vehicle.total_power_consumed += 4000
    vehicle.daily_stats = ev(rows(d20261006=13089, d20261008=1000)).daily_stats
    await tracker.async_process()
    assert store.data is not None and store.data["missed_wh"] == 3000
    assert "3000 Wh of driving was reported for days older" in caplog.text


class FakeStore:
    """In-memory stand-in for ``homeassistant.helpers.storage.Store``."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data
        self.saves = 0

    async def async_load(self) -> dict[str, Any] | None:
        return self.data

    async def async_save(self, data: dict[str, Any]) -> None:
        self.data = data
        self.saves += 1


class FakeStoreType:
    """Replaces ``Store``; ``FakeStoreType[...](hass, ...)`` returns a FakeStore."""

    def __class_getitem__(cls, item: object) -> type["FakeStoreType"]:
        return cls

    def __new__(cls, *args: object, **kwargs: object) -> FakeStore:  # type: ignore[misc]
        return FakeStore()


def ev(day_rows: dict[date, float] | None) -> Vehicle:
    vehicle = Vehicle(id="v1", name="test", model="EV6")
    vehicle.odometer = (50828.1, "km")
    vehicle.power_consumption_30d = 158
    vehicle.total_power_consumed = 959700
    vehicle.daily_stats = (
        [
            DailyDrivingStats(
                date=datetime(d.year, d.month, d.day),
                total_consumed=wh,
                distance=round(wh / 160),
            )
            for d, wh in day_rows.items()
        ]
        if day_rows is not None
        else None
    )
    return vehicle


def test_distance_rows_in_miles_are_converted() -> None:
    vehicle = ev(None)
    vehicle.daily_stats = [
        DailyDrivingStats(
            date=datetime(2026, 10, 6),
            total_consumed=100,
            distance=10,
            distance_unit="mi",
        )
    ]
    assert daily_distance_rows(vehicle) == {date(2026, 10, 6): pytest.approx(16.09344)}


async def test_tracker_saves_before_publishing() -> None:
    store = FakeStore()
    vehicle = ev(rows(d20261006=13089))
    tracker = DrivingMeterTracker(store, vehicle, lambda: NOW)
    await tracker.async_load()
    await tracker.async_process()

    published: list[float] = []

    def listener() -> None:
        assert tracker.meter is not None
        assert store.data is not None
        assert store.data["energy"]["total"] == tracker.meter.energy.total
        published.append(tracker.meter.energy.total)

    tracker.add_listener(listener)
    vehicle.daily_stats = ev(rows(d20261006=13089, d20261008=13238)).daily_stats
    vehicle.total_power_consumed += 13238
    await tracker.async_process()
    saves = store.saves
    await tracker.async_process()  # nothing changed
    assert published == [13238]
    assert store.saves == saves


async def test_tracker_resumes_from_store() -> None:
    store = FakeStore()
    tracker = DrivingMeterTracker(store, ev(rows(d20261006=13089)), lambda: NOW)
    await tracker.async_load()
    await tracker.async_process()

    vehicle = ev(rows(d20261006=13089, d20261008=13238))
    resumed = DrivingMeterTracker(store, vehicle, lambda: NOW)
    await resumed.async_load()
    await resumed.async_process()
    assert resumed.meter is not None
    assert resumed.meter.energy.total == 13238
    assert resumed.meter.distance.total == 83


async def test_sensor_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sensor_platform, "Store", FakeStoreType)
    vehicle = ev(rows(d20261006=13089))
    hass, config_entry = make_hass(make_coordinator(vehicle))
    config_entry.options = {CONF_LIFETIME_AVERAGE_CONSUMPTION: 18.5}
    created: list[Any] = []
    await sensor_platform.async_setup_entry(hass, config_entry, created.extend)

    by_type = {type(entity).__name__: entity for entity in created}
    meter = by_type["DrivingEnergySensor"]
    distance = by_type["DrivingDistanceSensor"]
    estimate = by_type["LifetimeDrivingEnergyEstimateSensor"]
    assert meter.state_class == SensorStateClass.TOTAL
    assert meter.native_value == 0
    assert meter.unique_id == "kia_uvo_v1_driving_energy"
    assert meter.native_unit_of_measurement == "Wh"
    assert distance.state_class == SensorStateClass.TOTAL
    assert distance.native_value == 0
    assert distance.unique_id == "kia_uvo_v1_driving_distance"
    assert distance.native_unit_of_measurement == "km"
    assert estimate.state_class is None
    assert estimate.native_value == pytest.approx(50828.1 * 185)

    window = next(
        entity
        for entity in created
        if getattr(entity, "entity_description", None) is not None
        and entity.entity_description.key == "total_power_consumed"
    )
    assert window.state_class is None


def test_backfill_starts_at_zero_before_first_state() -> None:
    meter = DrivingMeter()
    meter.update(
        rows(d20261005=26762, d20261006=25516, d20261008=19465),
        None,
        NOW,
    )
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
    meter.update(rows(d20261008=500), None, morning)
    backfill = backfill_statistics(meter.energy, TZ)
    assert backfill is not None
    # The last full hour before the meter's first state.
    assert backfill.rows == [(datetime(2026, 10, 8, 8, tzinfo=TZ), 500)]
    assert backfill.meter_start == datetime(2026, 10, 8, 9, tzinfo=TZ)


def test_backfill_needs_a_started_meter() -> None:
    assert backfill_statistics(DailyTotal(), TZ) is None


async def test_tracker_marks_backfilled_and_backdated() -> None:
    store = FakeStore()
    tracker = DrivingMeterTracker(store, ev(rows(d20261006=13089)), lambda: NOW)
    await tracker.async_load()
    await tracker.async_process()
    await tracker.async_mark_backfilled("distance")
    assert store.data is not None
    assert store.data["distance"]["backfilled"]
    assert not store.data["energy"]["backfilled"]
    assert store.data["energy"]["baseline_rows"] == {"2026-10-06": 13089}

    tracker._vehicle = ev(rows(d20261006=14089))
    await tracker.async_process()
    assert store.data["energy"]["late"] == {"2026-10-06": 1000}
    await tracker.async_mark_backdated("energy", {"2026-10-06": 1000})
    assert store.data["energy"]["late"] == {}
    assert store.data["energy"]["backdated"] == 1000


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


async def _started_sensors(
    monkeypatch: pytest.MonkeyPatch, vehicle: Vehicle, compiled: bool = True
) -> tuple[dict[str, Any], list[tuple[Any, list[Any]]], FakeRecorder]:
    """Set up the sensors as added to a hass with a recorder."""
    from homeassistant.components.recorder import statistics as recorder_statistics
    from homeassistant.helpers import recorder

    imported: list[tuple[Any, list[Any]]] = []
    monkeypatch.setattr(
        recorder_statistics,
        "async_import_statistics",
        lambda hass, metadata, stats: imported.append((metadata, list(stats))),
    )
    fake_recorder = FakeRecorder()
    monkeypatch.setattr(recorder, "get_instance", lambda hass: fake_recorder)
    monkeypatch.setattr(sensor_platform.dt_util, "now", lambda: NOW)
    monkeypatch.setattr(sensor_platform.dt_util, "get_default_time_zone", lambda: TZ)
    monkeypatch.setattr(sensor_platform, "Store", FakeStoreType)
    hass, config_entry = make_hass(make_coordinator(vehicle))
    config_entry.options = {}
    created: list[Any] = []
    await sensor_platform.async_setup_entry(hass, config_entry, created.extend)

    async def has_short_term_statistics(self: Any) -> bool:
        return compiled

    monkeypatch.setattr(
        sensor_platform.DrivingTotalSensor,
        "_async_has_short_term_statistics",
        has_short_term_statistics,
    )
    sensors = {
        type(e).__name__: e
        for e in created
        if type(e).__name__ in ("DrivingEnergySensor", "DrivingDistanceSensor")
    }
    for name, sensor in sensors.items():
        sensor.hass = MagicMock()
        sensor.hass.config.components = {"recorder"}
        sensor.entity_id = f"sensor.ev6_{name.lower()}"
        sensor.async_write_ha_state = MagicMock()
    monkeypatch.setattr(
        sensor_platform.DrivingTotalSensor,
        "unit_of_measurement",
        property(lambda self: "kWh" if self._quantity == "energy" else "km"),
    )
    return sensors, imported, fake_recorder


async def test_sensor_imports_backfill_once(monkeypatch: pytest.MonkeyPatch) -> None:
    vehicle = ev(rows(d20261005=26762, d20261006=13089))
    sensors, imported, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    meter = sensors["DrivingEnergySensor"]

    await meter._async_update_statistics()
    await meter._async_update_statistics()

    assert len(imported) == 1
    metadata, stats = imported[0]
    assert metadata["statistic_id"] == "sensor.ev6_drivingenergysensor"
    assert metadata["source"] == "recorder"
    assert metadata["unit_of_measurement"] == "kWh"
    assert [s["sum"] for s in stats] == [26.762, pytest.approx(39.851)]
    # The sums compiled since the start continue from the imported total.
    assert fake_recorder.adjusted == [
        (
            "sensor.ev6_drivingenergysensor",
            datetime(2026, 10, 8, 21, tzinfo=TZ),
            39851,
            "Wh",
        )
    ]

    await sensors["DrivingDistanceSensor"]._async_update_statistics()
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
    energy = sensors["DrivingEnergySensor"]
    await energy._async_update_statistics()
    assert imported == []
    assert fake_recorder.adjusted == []
    assert energy._tracker.meter is not None
    assert not energy._tracker.meter.energy.backfilled


async def test_sensor_books_late_change_on_its_day_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = ev(rows(d20261006=13089))
    sensors, _, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    energy = sensors["DrivingEnergySensor"]
    tracker = energy._tracker
    await energy._async_update_statistics()
    fake_recorder.adjusted.clear()

    tracker._vehicle = ev(rows(d20261006=14689, d20261008=500))
    await tracker.async_process()
    assert energy.native_value == 500
    await energy._async_update_statistics()
    await energy._async_update_statistics()

    assert fake_recorder.adjusted == [
        (
            "sensor.ev6_drivingenergysensor",
            datetime(2026, 10, 6, 12, tzinfo=TZ),
            1600,
            "Wh",
        )
    ]
    assert tracker.meter is not None
    assert tracker.meter.energy.late == {}
    assert energy.native_value == 500
    assert energy.extra_state_attributes["booked_on_earlier_days_wh"] == 1600
    assert tracker.meter.lifetime_estimate_wh(158.0) == pytest.approx(
        50828.1 * 158 + 2100
    )


async def test_sensor_marks_changes_done_only_once_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash before the commit must leave the change to be made again."""
    vehicle = ev(rows(d20261005=26762, d20261006=13089))
    sensors, imported, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    energy = sensors["DrivingEnergySensor"]
    tracker = energy._tracker
    assert tracker.meter is not None

    fake_recorder.commit = asyncio.get_running_loop().create_future()
    update = asyncio.create_task(energy._async_update_statistics())
    await asyncio.sleep(0)
    assert len(imported) == 1
    assert not tracker.meter.energy.backfilled
    fake_recorder.commit.set_result(None)
    await update
    assert tracker.meter.energy.backfilled

    tracker._vehicle = ev(rows(d20261005=26762, d20261006=14689))
    await tracker.async_process()
    fake_recorder.commit = asyncio.get_running_loop().create_future()
    update = asyncio.create_task(energy._async_update_statistics())
    await asyncio.sleep(0)
    assert fake_recorder.adjusted[-1][2] == 1600
    assert tracker.meter.energy.late == {"2026-10-06": 1600}
    fake_recorder.commit.set_result(None)
    await update
    assert tracker.meter.energy.late == {}
    assert tracker.meter.energy.backdated == 1600


async def test_sensor_without_recorder_keeps_late_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = ev(rows(d20261006=13089))
    sensors, _, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    energy = sensors["DrivingEnergySensor"]
    energy.hass.config.components = set()
    energy._tracker._vehicle = ev(rows(d20261006=14689))
    await energy._tracker.async_process()
    await energy._async_update_statistics()
    assert fake_recorder.adjusted == []
    assert energy._tracker.meter is not None
    assert energy._tracker.meter.energy.late == {"2026-10-06": 1600}
