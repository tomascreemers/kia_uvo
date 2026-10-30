"""Tests for the statistics-safe driving energy and distance meters."""

import asyncio
import copy
from datetime import date, datetime, time, timedelta
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

from homeassistant.components.sensor import SensorStateClass
from homeassistant.core import CoreState
from homeassistant.util import dt as dt_util
from hyundai_kia_connect_api import Vehicle
from hyundai_kia_connect_api.Vehicle import DailyDrivingStats
import pytest

from custom_components.kia_uvo import (
    driving_energy,
    number as number_platform,
    sensor as sensor_platform,
)
from custom_components.kia_uvo.driving_energy import (
    ENERGY_PARTS,
    ROW_RETENTION,
    DailyTotal,
    DrivingMeter,
    DrivingMeterTracker,
    LateChange,
    backfill_statistics,
    daily_distance_rows,
    daily_energy_rows,
    daily_part_rows,
)
from tests.conftest import make_coordinator, make_hass

TZ = ZoneInfo("Europe/Brussels")
NOW = datetime(2026, 10, 8, 21, 7, tzinfo=TZ)


def rows(**days: float) -> dict[date, float]:
    """Rows keyed like ``d20261006=13089``."""
    return {datetime.strptime(k[1:], "%Y%m%d").date(): v for k, v in days.items()}


def late(at: datetime = NOW, **days: float) -> list[LateChange]:
    """Late changes reported ``at``, keyed like ``rows``."""
    return [LateChange(day, value, at) for day, value in rows(**days).items()]


def daily_stats(energy_rows: dict[date, float]) -> list[DailyDrivingStats]:
    """The library's daily rows, at 160 Wh/km."""
    return [
        DailyDrivingStats(
            date=datetime(day.year, day.month, day.day),
            total_consumed=wh,
            engine_consumption=wh - round(wh * 0.04) - round(wh * 0.03),
            climate_consumption=round(wh * 0.04),
            onboard_electronics_consumption=round(wh * 0.03),
            battery_care_consumption=0,
            regenerated_energy=round(wh * 0.15),
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
        self.statistics = FakeStatistics()

    async def async_add_executor_job(self, target: Any, *args: Any) -> Any:
        return target(*args)

    async def async_block_till_done(self) -> None:
        if self.commit is not None:
            await self.commit

    def async_adjust_statistics(
        self, statistic_id: str, start: datetime, adjustment: float, unit: str
    ) -> None:
        self.adjusted.append((statistic_id, start, adjustment, unit))


class FakeStatistics:
    """A sensor's hourly statistics: every hour up to ``compiled``, but ``missing``.

    Stands in for the recorder's statistics queries; ``units`` are the units
    the recorder keeps, by statistic id.
    """

    def __init__(self) -> None:
        self.compiled = datetime(2027, 1, 1, tzinfo=dt_util.UTC)
        self.missing: set[datetime] = set()
        self.units: dict[str, str] = {}

    def last(self, hass: Any, number: int, statistic_id: str, *args: Any) -> Any:
        return {statistic_id: [{"start": self.compiled.timestamp(), "sum": 0.0}]}

    def during(
        self,
        hass: Any,
        start: datetime,
        end: datetime,
        ids: set[str],
        period: str,
        *args: Any,
    ) -> Any:
        (statistic_id,) = ids
        if period == "day":
            return {
                statistic_id: [{"start": start.timestamp(), "sum": 7.0, "state": 7.0}]
            }
        hours = []
        hour = start
        while hour < end and hour <= self.compiled:
            if hour not in self.missing:
                hours.append({"start": hour.timestamp(), "sum": 0.0, "state": 0.0})
            hour += timedelta(hours=1)
        return {statistic_id: hours}

    def metadata(self, hass: Any, *, statistic_ids: set[str]) -> Any:
        return {
            statistic_id: (1, {"unit_of_measurement": self.units[statistic_id]})
            for statistic_id in statistic_ids
            if statistic_id in self.units
        }


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


def test_reading_starts_at_the_baseline() -> None:
    meter = started_meter()
    assert meter.energy.reading == 39851
    meter.update(rows(d20261005=26762, d20261006=14089, d20261008=500), {}, NOW)
    # The late change to 6 Oct counts in the reading as soon as it arrives.
    assert meter.energy.reading == 41351


def test_late_past_day_is_kept_apart_from_today() -> None:
    meter = started_meter()
    # Recorded 2026-10-08: the 6 Oct return trip arrived 39 h late,
    # together with the first drive of 8 Oct.
    meter.update(rows(d20261005=26762, d20261006=25516, d20261008=13238), {}, NOW)
    assert meter.energy.total == 13238
    assert meter.energy.late == late(d20261006=12427)
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
    assert meter.energy.late == late(saturday, d20261008=7155)
    assert meter.distance.total == 2
    assert meter.distance.late == late(saturday, d20261008=42)
    assert meter.energy.late_hour(date(2026, 10, 8), TZ) == datetime(
        2026, 10, 8, 23, tzinfo=TZ
    )


def test_late_change_before_the_start_goes_to_the_backfilled_hour() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27000, d20261006=13089), {}, NOW)
    assert meter.energy.late == late(d20261005=238)
    assert meter.energy.late_hour(date(2026, 10, 5), TZ) == datetime(
        2026, 10, 5, 12, tzinfo=TZ
    )


def test_new_empty_row_is_not_late() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=26762, d20261006=13089, d20261007=0), {}, NOW)
    assert meter.energy.late == []
    assert date(2026, 10, 7) in meter.energy.rows


def test_mark_backdated_moves_late_into_backdated() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27000, d20261006=14089), {}, NOW)
    applied = list(meter.energy.late)
    # More arrives for the same day before the first part is marked.
    later = NOW + timedelta(minutes=30)
    meter.update(rows(d20261005=27100, d20261006=14089), {}, later)
    reading = meter.energy.reading
    meter.energy.mark_backdated(applied)
    assert meter.energy.late == late(later, d20261005=100)
    assert meter.energy.backdated == 1238
    assert meter.energy.measured == 1338
    assert meter.energy.reading == reading
    meter.energy.mark_backdated(late(later, d20261005=100))
    assert meter.energy.late == []


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
    assert meter.energy.late == late(d20261005=-762)
    assert meter.energy.total == 400
    assert meter.energy.reading == 39851 - 762 + 400


def test_rows_are_forgotten_by_age_and_old_rows_ignored() -> None:
    meter = started_meter()
    later = datetime(2026, 11, 12, 9, tzinfo=TZ)
    meter.update(rows(d20261110=5000), {}, later)
    assert meter.energy.rows == rows(d20261110=5000)
    assert meter.energy.late == late(later, d20261110=5000)
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
    assert meter.lifetime_estimate_wh() == pytest.approx(50828.1 * 158 + measured)
    meter.average_wh_per_km = 185.0
    assert meter.lifetime_estimate_wh() == pytest.approx(50828.1 * 185 + measured)
    assert DrivingMeter(average_wh_per_km=185.0).lifetime_estimate_wh() is None


def test_meter_saved_before_the_estimate_loads() -> None:
    data = started_meter().as_dict()
    del data["odometer_at_start_km"], data["api_wh_per_km_at_start"]
    del data["average_wh_per_km"]
    meter = DrivingMeter.from_dict(data)
    meter.average_wh_per_km = 185.0
    # The odometer at its start is unknown, so it gets no estimate.
    assert meter.lifetime_estimate_wh() is None


def test_round_trip() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=27762, d20261006=13089, d20261008=1000), {}, NOW)
    meter.update_parts({"energy_climate": rows(d20261006=400)}, NOW)
    meter.update(rows(d20261005=28762, d20261006=13089, d20261008=1000), {}, NOW)
    meter.energy.mark_backdated(meter.energy.late[:1])
    assert DrivingMeter.from_dict(meter.as_dict()) == meter


def test_stored_format() -> None:
    """Pin the stored format: changing it would need a storage migration."""
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
            "late": [
                {
                    "day": "2026-10-05",
                    "value": 1000.0,
                    "reported_at": "2026-10-08T21:07:00+02:00",
                }
            ],
            "backdated": 0.0,
        },
        "distance": {
            "total": 0.0,
            "rows": {},
            "started_at": None,
            "baseline_rows": {},
            "backfilled": False,
            "late": [],
            "backdated": 0.0,
        },
        **{part: DailyTotal().as_dict() for part in ENERGY_PARTS},
        "odometer_at_start_km": None,
        "api_wh_per_km_at_start": None,
        "average_wh_per_km": None,
        "window_wh": 963025.0,
        "window_carry_wh": 0.0,
        "missed_wh": 0.0,
    }


# The energy parts


CONSUMED_PARTS = (
    "energy_motor",
    "energy_climate",
    "energy_electronics",
    "energy_battery_care",
)


def day(
    year: int,
    month: int,
    day_: int,
    motor: int,
    climate: int,
    electronics: int,
    battery_care: int,
    regenerated: int,
) -> DailyDrivingStats:
    return DailyDrivingStats(
        date=datetime(year, month, day_),
        total_consumed=motor + climate + electronics + battery_care,
        engine_consumption=motor,
        climate_consumption=climate,
        onboard_electronics_consumption=electronics,
        battery_care_consumption=battery_care,
        regenerated_energy=regenerated,
    )


def assert_sums(parts: list[dict[date, float]], total: dict[date, float]) -> None:
    assert {key: sum(part.get(key, 0.0) for part in parts) for key in total} == total


def by_day(changes: list[LateChange]) -> dict[date, float]:
    """The late changes added up per day."""
    days: dict[date, float] = {}
    for change in changes:
        days[change.day] = days.get(change.day, 0.0) + change.value
    return days


def test_consumed_parts_add_up_to_the_energy() -> None:
    # Rows of 2026-10-17 to 19 from the 30-day response, with each day
    # growing trip by trip and 18 Oct revised a day late.
    vehicle = ev({})
    meter = DrivingMeter()

    def poll(now: datetime, *days: DailyDrivingStats) -> None:
        vehicle.daily_stats = list(days)
        meter.update(daily_energy_rows(vehicle), {}, now)
        meter.update_parts(daily_part_rows(vehicle), now)

    oct17 = day(2026, 10, 17, 9694, 1098, 430, 0, 3622)
    poll(datetime(2026, 10, 18, 8, tzinfo=TZ), oct17)
    poll(
        datetime(2026, 10, 18, 20, tzinfo=TZ),
        oct17,
        day(2026, 10, 18, 4000, 500, 300, 0, 2000),
    )
    revised = datetime(2026, 10, 19, 21, tzinfo=TZ)
    poll(
        revised,
        oct17,
        day(2026, 10, 18, 6056, 819, 490, 0, 2710),
        day(2026, 10, 19, 25328, 1226, 680, 0, 3312),
    )

    parts = [meter.daily_total(part) for part in CONSUMED_PARTS]
    assert sum(part.total for part in parts) == meter.energy.total
    assert sum(part.measured for part in parts) == meter.energy.measured
    assert_sums([by_day(part.late) for part in parts], by_day(meter.energy.late))
    assert_sums([part.baseline_rows for part in parts], meter.energy.baseline_rows)
    assert meter.energy.total == 4800 + 27234
    assert meter.energy.late == late(revised, d20261018=2565)
    assert meter.energy_regenerated.total == 2000 + 3312
    assert meter.energy_regenerated.late == late(revised, d20261018=710)


def test_part_rows_skip_days_without_the_part() -> None:
    vehicle = ev({})
    vehicle.daily_stats = [
        day(2026, 10, 6, 900, 50, 40, 10, 200),
        DailyDrivingStats(date=datetime(2026, 10, 7), total_consumed=100),
    ]
    assert daily_part_rows(vehicle) == {
        "energy_motor": rows(d20261006=900),
        "energy_climate": rows(d20261006=50),
        "energy_electronics": rows(d20261006=40),
        "energy_battery_care": rows(d20261006=10),
        "energy_regenerated": rows(d20261006=200),
    }


async def test_parts_start_on_their_own_next_to_a_running_meter() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=26762, d20261006=13089, d20261008=1000), {}, NOW)
    store = FakeStore()
    # Stored before the parts existed.
    store.data = {
        key: value for key, value in meter.as_dict().items() if key not in ENERGY_PARTS
    }
    vehicle = ev(rows(d20261005=26762, d20261006=13089, d20261008=1500))
    tracker = await DrivingMeterTracker.async_load(store, vehicle, lambda: NOW)
    await tracker.async_process()
    assert tracker.meter.energy.total == 1500
    assert tracker.meter.energy_motor.initialized
    assert tracker.meter.energy_motor.total == 0
    assert tracker.meter.energy_climate.baseline_rows == rows(
        d20261005=1070, d20261006=524, d20261008=60
    )


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
    assert backfill.rows[-4:] == [
        (datetime(2026, 10, 5, 12, tzinfo=TZ), 26762),
        (datetime(2026, 10, 6, 12, tzinfo=TZ), 52278),
        (datetime(2026, 10, 7, 12, tzinfo=TZ), 52278),
        (datetime(2026, 10, 8, 12, tzinfo=TZ), 71743),
    ]
    assert backfill.total == 71743
    assert backfill.meter_start == datetime(2026, 10, 8, 21, tzinfo=TZ)


def test_backfill_has_a_row_for_every_day_a_late_change_can_reach() -> None:
    meter = DrivingMeter()
    meter.update(rows(d20261006=25516), {}, NOW)
    backfill = backfill_statistics(meter.energy, TZ)
    assert backfill is not None
    oldest = date(2026, 10, 8) - ROW_RETENTION
    days = [oldest + timedelta(days=n) for n in range(ROW_RETENTION.days + 1)]
    assert [hour for hour, _ in backfill.rows] == [
        datetime.combine(day, time(12), TZ) for day in days
    ]
    assert backfill.rows[0][1] == 0
    # A change to the undriven 10-07 lands on its own row, not on 10-08's.
    hour = meter.energy.late_hour(date(2026, 10, 7), TZ)
    assert (hour, 25516) in backfill.rows


def test_backfill_moves_start_day_before_a_morning_install() -> None:
    meter = DrivingMeter()
    morning = datetime(2026, 10, 8, 9, 30, tzinfo=TZ)
    meter.update(rows(d20261008=500), {}, morning)
    backfill = backfill_statistics(meter.energy, TZ)
    assert backfill is not None
    # The last full hour before the meter's first state.
    assert backfill.rows[-1] == (datetime(2026, 10, 8, 8, tzinfo=TZ), 500)
    assert backfill.rows[-2] == (datetime(2026, 10, 7, 12, tzinfo=TZ), 0)
    assert backfill.meter_start == datetime(2026, 10, 8, 9, tzinfo=TZ)


def test_backfill_merges_days_that_share_an_hour() -> None:
    meter = DrivingMeter()
    morning = datetime(2026, 10, 8, 9, 30, tzinfo=TZ)
    # A row dated after the local date, from a car in another time zone.
    meter.update(rows(d20261007=300, d20261008=500, d20261009=200), {}, morning)
    backfill = backfill_statistics(meter.energy, TZ)
    assert backfill is not None
    assert backfill.rows[-2:] == [
        (datetime(2026, 10, 7, 12, tzinfo=TZ), 300),
        (datetime(2026, 10, 8, 8, tzinfo=TZ), 1000),
    ]
    assert backfill.total == 1000


def test_statistics_hours_are_utc_hours_half_an_hour_off_utc() -> None:
    kolkata = ZoneInfo("Asia/Kolkata")
    meter = DrivingMeter()
    meter.update(
        rows(d20261006=25516), {}, datetime(2026, 10, 8, 21, 7, tzinfo=kolkata)
    )
    backfill = backfill_statistics(meter.energy, kolkata)
    assert backfill is not None
    assert backfill.meter_start == datetime(2026, 10, 8, 15, tzinfo=dt_util.UTC)
    assert all(hour.astimezone(dt_util.UTC).minute == 0 for hour, _ in backfill.rows)
    assert meter.energy.late_hour(date(2026, 10, 9), kolkata) == datetime(
        2026, 10, 9, 17, tzinfo=dt_util.UTC
    )
    reported = datetime(2026, 10, 9, 10, 22, tzinfo=kolkata)
    assert sensor_platform._short_term_period(reported) == datetime(
        2026, 10, 9, 4, 50, tzinfo=dt_util.UTC
    )


def test_backfill_ends_before_a_start_just_after_spring_forward() -> None:
    meter = DrivingMeter()
    meter.update(rows(d20260329=500), {}, datetime(2026, 3, 29, 3, 20, tzinfo=TZ))
    backfill = backfill_statistics(meter.energy, TZ)
    assert backfill is not None
    assert backfill.rows[-1] == (datetime(2026, 3, 29, 0, tzinfo=dt_util.UTC), 500)
    assert backfill.meter_start == datetime(2026, 3, 29, 1, tzinfo=dt_util.UTC)


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
    assert store.data["energy"]["late"] == [
        {"day": "2026-10-06", "value": 1000, "reported_at": NOW.isoformat()}
    ]
    await tracker.async_mark_backdated("energy", late(d20261006=1000))
    assert store.data["energy"]["late"] == []
    assert store.data["energy"]["backdated"] == 1000


# The sensors


async def _set_up(
    monkeypatch: pytest.MonkeyPatch, vehicle: Vehicle, store: FakeStore | None = None
) -> tuple[Any, Any, Any, list[Any]]:
    """Set the entry up as ``__init__`` does, then its sensor platform.

    Return hass, the config entry, the coordinator and the sensors.
    """
    monkeypatch.setattr(driving_energy, "Store", lambda *args: store or FakeStore())
    monkeypatch.setattr(dt_util, "now", lambda: NOW)
    coordinator = make_coordinator(vehicle)
    hass, config_entry = make_hass(coordinator)
    coordinator.driving_meters = await driving_energy.async_setup_driving_meters(
        hass, config_entry, coordinator
    )
    created: list[Any] = []
    await sensor_platform.async_setup_entry(hass, config_entry, created.extend)
    return hass, config_entry, coordinator, created


async def _set_up_sensors(
    monkeypatch: pytest.MonkeyPatch, vehicle: Vehicle
) -> dict[str, Any]:
    """Return the sensors set up for ``vehicle``, by unique id less its prefix."""
    *_, created = await _set_up(monkeypatch, vehicle)
    return {entity.unique_id.removeprefix("kia_uvo_v1_"): entity for entity in created}


async def test_sensor_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    sensors = await _set_up_sensors(monkeypatch, ev(rows(d20261006=13089)))
    energy = sensors["driving_energy"]
    distance = sensors["driving_distance"]
    assert energy.state_class == SensorStateClass.TOTAL
    # The day before the start, which the backfill leads up to.
    assert energy.native_value == 13089
    assert energy.unique_id == "kia_uvo_v1_driving_energy"
    assert energy.native_unit_of_measurement == "Wh"
    assert distance.state_class == SensorStateClass.TOTAL
    assert distance.native_value == 82
    assert distance.unique_id == "kia_uvo_v1_driving_distance"
    assert distance.native_unit_of_measurement == "km"


async def test_meter_work_runs_as_tasks_of_the_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload waits for them, and none starts before Home Assistant has."""
    hass, config_entry, coordinator, created = await _set_up(
        monkeypatch, ev(rows(d20261006=13089))
    )

    schedule_process = coordinator.async_add_listener.call_args.args[0]
    coordinator.last_update_success = False
    schedule_process()
    config_entry.async_create_task.assert_not_called()
    coordinator.last_update_success = True
    schedule_process()
    energy = next(e for e in created if e.unique_id == "kia_uvo_v1_driving_energy")
    energy.hass = hass
    energy._schedule_statistics_update()
    hass.state = CoreState.running
    energy._schedule_statistics_update()

    names = [call.args[2] for call in config_entry.async_create_task.call_args_list]
    assert names == [
        "kia_uvo driving meters v1",
        f"kia_uvo {energy.entity_id} statistics",
    ]
    for call in config_entry.async_create_task.call_args_list:
        call.args[1].close()
    hass.async_create_task.assert_not_called()


async def test_tracker_outlives_a_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    vehicle = ev(rows(d20261006=13089))
    hass, config_entry, coordinator, _ = await _set_up(monkeypatch, vehicle)
    tracker = coordinator.driving_meters["v1"]
    reloaded = make_coordinator(ev(rows(d20261006=14089)))
    found = await driving_energy.async_setup_driving_meters(
        hass, config_entry, reloaded
    )
    assert found["v1"] is tracker
    assert tracker.vehicle is reloaded.vehicle_manager.vehicles["v1"]
    assert tracker.meter.energy.late == late(d20261006=1000)


async def test_unreadable_meter_store_leaves_the_other_sensors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeStore()
    store.data = {"energy": {}}
    hass, *_, created = await _set_up(monkeypatch, ev(rows(d20261006=13089)), store)
    keys = {entity.unique_id.removeprefix("kia_uvo_v1_") for entity in created}
    assert "total_power_consumed" in keys
    assert "driving_energy" not in keys
    assert store.data == {"energy": {}}
    assert hass.data[driving_energy.DRIVING_METERS] == {}


async def test_rolling_window_sensors_have_no_state_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = ev(rows(d20261006=13089))
    vehicle.total_power_regenerated = 1000
    sensors = await _set_up_sensors(monkeypatch, vehicle)
    assert sensors["total_power_consumed"].state_class is None
    assert sensors["total_power_regenerated"].state_class is None


async def test_lifetime_estimate_sensor(monkeypatch: pytest.MonkeyPatch) -> None:
    vehicle = ev(rows(d20261006=13089))
    vehicle.power_consumption_30d = 158
    hass, config_entry, coordinator, created = await _set_up(monkeypatch, vehicle)
    estimate = next(
        e for e in created if e.unique_id.endswith("lifetime_driving_energy_estimate")
    )
    assert estimate.state_class is None
    assert estimate.native_value == pytest.approx(50828.1 * 158)
    assert estimate.extra_state_attributes["average_source"] == "api_30_day_average"

    numbers: list[Any] = []
    await number_platform.async_setup_entry(hass, config_entry, numbers.extend)
    (average,) = (n for n in numbers if n.unique_id.endswith("average_consumption"))
    assert average.unique_id == "kia_uvo_v1_lifetime_average_consumption"
    assert average.available
    assert average.native_value == 15.8
    await average.async_set_native_value(18.5)

    tracker = coordinator.driving_meters["v1"]
    assert tracker.meter.average_wh_per_km == 185
    assert average.native_value == 18.5
    assert estimate.native_value == pytest.approx(50828.1 * 185)
    assert estimate.extra_state_attributes == {
        "measured_since": "2026-10-08",
        "odometer_at_start_km": 50828.1,
        "average_consumption_kwh_per_100km": 18.5,
        "average_source": "configured",
    }


async def test_each_car_keeps_its_own_average(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = ev(rows(d20261006=13089)), ev(rows(d20261006=9000))
    second.id = "v2"
    monkeypatch.setattr(driving_energy, "Store", lambda *args: FakeStore())
    monkeypatch.setattr(dt_util, "now", lambda: NOW)
    coordinator = make_coordinator(first, second)
    hass, config_entry = make_hass(coordinator)
    coordinator.driving_meters = await driving_energy.async_setup_driving_meters(
        hass, config_entry, coordinator
    )
    await coordinator.driving_meters["v1"].async_set_average(185)
    assert coordinator.driving_meters["v2"].meter.average_wh_per_km is None


async def test_lifetime_estimate_sensor_without_an_average(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dt_util, "now", lambda: NOW)
    sensors = await _set_up_sensors(monkeypatch, ev(rows(d20261006=13089)))
    estimate = sensors["lifetime_driving_energy_estimate"]
    assert estimate.native_value is None
    assert estimate.extra_state_attributes["average_source"] is None


async def test_energy_part_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    sensors = await _set_up_sensors(monkeypatch, ev(rows(d20261006=13089)))
    parts = {
        key: sensor
        for key, sensor in sensors.items()
        if key.startswith("driving_energy_")
    }
    assert sorted(parts) == [
        "driving_energy_battery_care",
        "driving_energy_climate",
        "driving_energy_electronics",
        "driving_energy_motor",
        "driving_energy_regenerated",
    ]
    for key, part in parts.items():
        assert part.translation_key == key
        assert part.state_class == SensorStateClass.TOTAL
        assert part.device_class == "energy"
        assert part.native_unit_of_measurement == "Wh"
    # Like the driving energy, they start at the days before the meter, and
    # all but the regenerated energy add up to it.
    regenerated = parts.pop("driving_energy_regenerated")
    assert regenerated.native_value == 1963
    assert sum(part.native_value for part in parts.values()) == 13089
    assert sensors["driving_energy"].native_value == 13089


async def _started_sensors(
    monkeypatch: pytest.MonkeyPatch,
    vehicle: Vehicle,
    compiled: datetime | None = NOW,
) -> tuple[dict[str, Any], list[tuple[Any, list[Any]]], FakeRecorder]:
    """Set up the sensors as added to a hass with a recorder.

    ``compiled`` is the start of the last short-term statistics.
    """
    imported: list[tuple[Any, list[Any]]] = []
    monkeypatch.setattr(
        sensor_platform,
        "async_import_statistics",
        lambda hass, metadata, stats: imported.append((metadata, list(stats))),
    )
    fake_recorder = FakeRecorder()
    monkeypatch.setattr(sensor_platform, "get_instance", lambda hass: fake_recorder)
    statistics = fake_recorder.statistics
    monkeypatch.setattr(sensor_platform, "get_last_statistics", statistics.last)
    monkeypatch.setattr(sensor_platform, "statistics_during_period", statistics.during)
    monkeypatch.setattr(sensor_platform, "get_metadata", statistics.metadata)
    monkeypatch.setattr(dt_util, "now", lambda: NOW)
    monkeypatch.setattr(dt_util, "get_default_time_zone", lambda: TZ)
    sensors = {
        key: sensor
        for key, sensor in (await _set_up_sensors(monkeypatch, vehicle)).items()
        if isinstance(sensor, sensor_platform.DrivingMeterSensor)
    }

    compile_up_to(monkeypatch, compiled)
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


def compile_up_to(monkeypatch: pytest.MonkeyPatch, start: datetime | None) -> None:
    """Have the recorder's last short-term statistics start at ``start``."""

    async def last_short_term_start(self: Any) -> float | None:
        return None if start is None else start.timestamp()

    monkeypatch.setattr(
        sensor_platform.DrivingMeterSensor,
        "_async_last_short_term_start",
        last_short_term_start,
    )


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
    assert [s["sum"] for s in stats][-5:] == [0, 26.762] + [pytest.approx(39.851)] * 3
    # The states lead up to the sensor's first state.
    assert [s["state"] for s in stats] == [s["sum"] for s in stats]
    assert energy.native_value == 39851
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
    assert [s["sum"] for s in stats][-4:] == [167, 249, 249, 249]

    await sensors["driving_energy_climate"]._async_update_statistics()
    metadata, stats = imported[2]
    assert metadata["statistic_id"] == "sensor.ev6_driving_energy_climate"
    assert metadata["unit_class"] == "energy"
    assert [s["sum"] for s in stats][-4:] == [1.07] + [pytest.approx(1.594)] * 3


async def test_sensor_waits_for_short_term_statistics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = ev(rows(d20261005=26762, d20261006=13089))
    sensors, imported, fake_recorder = await _started_sensors(
        monkeypatch, vehicle, compiled=None
    )
    energy = sensors["driving_energy"]
    await energy._async_update_statistics()
    assert imported == []
    assert fake_recorder.adjusted == []
    assert not energy._tracker.meter.energy.backfilled


async def test_sensor_moves_late_change_to_its_day_once(
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
    # The state counts the late change at once; the recorder books it now.
    assert energy.native_value == 15189
    await energy._async_update_statistics()
    await energy._async_update_statistics()

    # Then it moves from the 5 minutes it was reported in to its day.
    assert fake_recorder.adjusted == [
        (
            "sensor.ev6_driving_energy",
            datetime(2026, 10, 6, 12, tzinfo=TZ),
            1600,
            "Wh",
        ),
        (
            "sensor.ev6_driving_energy",
            datetime(2026, 10, 8, 21, 5, tzinfo=TZ),
            -1600,
            "Wh",
        ),
    ]
    assert tracker.meter.energy.late == []
    assert energy.native_value == 15189
    assert energy.extra_state_attributes == {
        "measured_since": "2026-10-08",
        "booked_on_earlier_days_wh": 1600,
    }
    assert tracker.meter.energy.measured == 2100
    await tracker.async_set_average(158.0)
    assert tracker.meter.lifetime_estimate_wh() == pytest.approx(50828.1 * 158 + 2100)


async def test_sensor_moves_late_change_once_its_period_is_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vehicle = ev(rows(d20261006=13089))
    sensors, _, fake_recorder = await _started_sensors(
        monkeypatch, vehicle, compiled=datetime(2026, 10, 8, 21, tzinfo=TZ)
    )
    energy = sensors["driving_energy"]
    await energy._async_update_statistics()
    fake_recorder.adjusted.clear()

    # Reported at 21:07: the recorder books it in the 21:05 period.
    vehicle.daily_stats = daily_stats(rows(d20261006=14689))
    await energy._tracker.async_process()
    await energy._async_update_statistics()
    assert fake_recorder.adjusted == []
    assert energy._tracker.meter.energy.late == late(d20261006=1600)

    compile_up_to(monkeypatch, datetime(2026, 10, 8, 21, 5, tzinfo=TZ))
    await energy._async_update_statistics()
    assert [adjustment[2] for adjustment in fake_recorder.adjusted] == [1600, -1600]
    assert energy._tracker.meter.energy.late == []


LATER = datetime(2026, 10, 9, 8, 2, tzinfo=TZ)


async def _report_late_on_the_start_day(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, FakeRecorder]:
    """Return the energy sensor, backfilled, with 500 Wh reported late for 10-08."""
    vehicle = ev(rows(d20261006=13089))
    sensors, _, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    energy = sensors["driving_energy"]
    await energy._async_update_statistics()
    fake_recorder.adjusted.clear()
    energy._tracker._now = lambda: LATER
    vehicle.daily_stats = daily_stats(rows(d20261006=13089, d20261008=500))
    await energy._tracker.async_process()
    assert energy._tracker.meter.energy.late == late(LATER, d20261008=500)
    compile_up_to(monkeypatch, LATER)
    return energy, fake_recorder


def utc(day: int, hour: int) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=dt_util.UTC)


async def test_late_change_books_on_the_last_compiled_hour_of_its_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy, fake_recorder = await _report_late_on_the_start_day(monkeypatch)
    # Unavailable from 22:00 local: no 22:00 and 23:00 rows.
    fake_recorder.statistics.missing = {utc(8, 20), utc(8, 21)}
    await energy._async_update_statistics()
    assert [(a[1], a[2]) for a in fake_recorder.adjusted] == [
        (utc(8, 19), 500),
        (datetime(2026, 10, 9, 8, tzinfo=TZ), -500),
    ]
    assert energy._tracker.meter.energy.late == []


async def test_late_change_to_a_day_without_statistics_gets_a_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy, fake_recorder = await _report_late_on_the_start_day(monkeypatch)
    statistics = fake_recorder.statistics
    statistics.missing = {utc(7, 22) + timedelta(hours=n) for n in range(24)}
    statistics.units["sensor.ev6_driving_energy"] = "Wh"
    imported: list[Any] = []
    monkeypatch.setattr(
        sensor_platform,
        "async_import_statistics",
        lambda hass, metadata, stats: imported.append((metadata, list(stats))),
    )
    await energy._async_update_statistics()
    ((metadata, stats),) = imported
    assert metadata["unit_of_measurement"] == "Wh"
    # 23:00 local, carrying the sum and state of the last row before it.
    assert stats == [{"start": utc(8, 21), "state": 7.0, "sum": 7.0}]
    assert fake_recorder.adjusted[0][1:3] == (utc(8, 21), 500)


async def test_late_change_waits_until_its_hour_is_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy, fake_recorder = await _report_late_on_the_start_day(monkeypatch)
    fake_recorder.statistics.compiled = utc(8, 20)
    await energy._async_update_statistics()
    assert fake_recorder.adjusted == []
    assert energy._tracker.meter.energy.late == late(LATER, d20261008=500)


async def test_reloaded_sensor_waits_for_its_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """However long the recorder takes, the work is done once."""
    vehicle = ev(rows(d20261005=26762, d20261006=13089))
    sensors, imported, fake_recorder = await _started_sensors(monkeypatch, vehicle)
    old = sensors["driving_energy"]
    new = sensor_platform.DrivingMeterSensor(
        old.coordinator, MagicMock(), vehicle, old._tracker, old.entity_description
    )
    new.hass, new.entity_id = old.hass, old.entity_id

    fake_recorder.commit = asyncio.get_running_loop().create_future()
    first = asyncio.create_task(old._async_update_statistics())
    await asyncio.sleep(0)
    second = asyncio.create_task(new._async_update_statistics())
    await asyncio.sleep(0)
    fake_recorder.commit.set_result(None)
    await asyncio.gather(first, second)
    assert len(imported) == 1
    assert len(fake_recorder.adjusted) == 1
    assert old._tracker.meter.energy.backfilled


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
    assert [adjustment[2] for adjustment in fake_recorder.adjusted[-2:]] == [
        1600,
        -1600,
    ]
    assert tracker.meter.energy.late == late(d20261006=1600)
    fake_recorder.commit.set_result(None)
    await update
    assert tracker.meter.energy.late == []
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
    assert energy._tracker.meter.energy.late == late(d20261006=1600)
