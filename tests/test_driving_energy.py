"""Tests for the statistics-safe driving energy meter."""

from datetime import date, datetime
from typing import Any

import pytest
from homeassistant.components.sensor import SensorStateClass
from hyundai_kia_connect_api import Vehicle
from hyundai_kia_connect_api.Vehicle import DailyDrivingStats

from custom_components.kia_uvo import sensor as sensor_platform
from custom_components.kia_uvo.const import CONF_LIFETIME_AVERAGE_CONSUMPTION
from custom_components.kia_uvo.driving_energy import (
    DrivingEnergyMeter,
    DrivingEnergyTracker,
)
from tests.conftest import make_coordinator, make_hass

TODAY = date(2026, 10, 8)


def rows(**days: float) -> dict[date, float]:
    """Rows keyed like ``d20261006=13089``."""
    return {datetime.strptime(k[1:], "%Y%m%d").date(): v for k, v in days.items()}


def started_meter() -> DrivingEnergyMeter:
    meter = DrivingEnergyMeter()
    meter.update(rows(d20261005=26762, d20261006=13089), TODAY, 50828.1, 158.0)
    return meter


def test_first_rows_only_set_the_baseline() -> None:
    meter = started_meter()
    assert meter.initialized
    assert meter.total_wh == 0
    assert meter.odometer_at_start_km == 50828.1


def test_growing_today_and_late_past_day_are_both_counted() -> None:
    meter = started_meter()
    # Recorded 2026-10-08: the 6 Oct return trip arrived 39 h late,
    # together with the first drive of 8 Oct.
    meter.update(rows(d20261005=26762, d20261006=25516, d20261008=13238), TODAY)
    assert meter.total_wh == 12427 + 13238
    meter.update(rows(d20261005=26762, d20261006=25516, d20261008=19465), TODAY)
    assert meter.total_wh == 31892


@pytest.mark.parametrize("empty", [None, {}])
def test_empty_response_changes_nothing(empty: dict[date, float] | None) -> None:
    meter = started_meter()
    assert not meter.update(empty, TODAY)
    meter.update(rows(d20261005=26762, d20261006=13089), TODAY)
    assert meter.total_wh == 0


def test_row_missing_from_one_response_is_not_counted_again() -> None:
    meter = started_meter()
    meter.update(rows(d20261006=13089), TODAY)
    meter.update(rows(d20261005=26762, d20261006=13089), TODAY)
    assert meter.total_wh == 0


def test_downward_correction_is_subtracted() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=26000, d20261006=13089), TODAY)
    assert meter.total_wh == -762


def test_rows_are_forgotten_by_age_and_old_rows_ignored() -> None:
    meter = started_meter()
    later = date(2026, 11, 12)
    meter.update(rows(d20261110=5000), later)
    assert set(meter.rows) == {"2026-11-10"}
    assert meter.total_wh == 5000
    # A row older than the retention window must never be counted.
    meter.update(rows(d20261005=26762, d20261110=5000), later)
    assert meter.total_wh == 5000


def test_lifetime_estimate() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=26762, d20261006=13089, d20261008=1000), TODAY)
    assert meter.lifetime_estimate_wh(185.0) == pytest.approx(50828.1 * 185 + 1000)
    assert meter.lifetime_estimate_wh(None) == pytest.approx(50828.1 * 158 + 1000)
    assert DrivingEnergyMeter().lifetime_estimate_wh(185.0) is None


def test_round_trip() -> None:
    meter = started_meter()
    meter.update(rows(d20261005=26762, d20261006=13089, d20261008=1000), TODAY)
    assert DrivingEnergyMeter.from_dict(meter.as_dict()) == meter


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
            DailyDrivingStats(date=datetime(d.year, d.month, d.day), total_consumed=wh)
            for d, wh in day_rows.items()
        ]
        if day_rows is not None
        else None
    )
    return vehicle


async def test_tracker_saves_before_publishing() -> None:
    store = FakeStore()
    vehicle = ev(rows(d20261006=13089))
    tracker = DrivingEnergyTracker(store, vehicle, lambda: TODAY)
    await tracker.async_load()
    await tracker.async_process()

    published: list[float] = []

    def listener() -> None:
        assert tracker.meter is not None
        assert store.data is not None
        assert store.data["total_wh"] == tracker.meter.total_wh
        published.append(tracker.meter.total_wh)

    tracker.add_listener(listener)
    vehicle.daily_stats = ev(rows(d20261006=13089, d20261008=13238)).daily_stats
    await tracker.async_process()
    saves = store.saves
    await tracker.async_process()  # nothing changed
    assert published == [13238]
    assert store.saves == saves


async def test_tracker_resumes_from_store() -> None:
    store = FakeStore()
    tracker = DrivingEnergyTracker(store, ev(rows(d20261006=13089)), lambda: TODAY)
    await tracker.async_load()
    await tracker.async_process()

    vehicle = ev(rows(d20261006=13089, d20261008=13238))
    resumed = DrivingEnergyTracker(store, vehicle, lambda: TODAY)
    await resumed.async_load()
    await resumed.async_process()
    assert resumed.meter is not None
    assert resumed.meter.total_wh == 13238


async def test_sensor_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sensor_platform, "Store", FakeStoreType)
    vehicle = ev(rows(d20261006=13089))
    hass, config_entry = make_hass(make_coordinator(vehicle))
    config_entry.options = {CONF_LIFETIME_AVERAGE_CONSUMPTION: 18.5}
    created: list[Any] = []
    await sensor_platform.async_setup_entry(hass, config_entry, created.extend)

    by_type = {type(entity).__name__: entity for entity in created}
    meter = by_type["DrivingEnergySensor"]
    estimate = by_type["LifetimeDrivingEnergyEstimateSensor"]
    assert meter.state_class == SensorStateClass.TOTAL
    assert meter.native_value == 0
    assert estimate.state_class is None
    assert estimate.native_value == pytest.approx(50828.1 * 185)

    window = next(
        entity
        for entity in created
        if getattr(entity, "entity_description", None) is not None
        and entity.entity_description.key == "total_power_consumed"
    )
    assert window.state_class is None
