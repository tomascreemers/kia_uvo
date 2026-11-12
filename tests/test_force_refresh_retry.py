"""Tests that a failed force refresh is not retried before reading the cache."""

from types import SimpleNamespace
from typing import Any

from homeassistant.helpers.update_coordinator import UpdateFailed
import pytest

from custom_components.kia_uvo.coordinator import HyundaiKiaConnectDataUpdateCoordinator


class FakeManager:
    """Records the per-vehicle calls and fails the ones it is told to."""

    def __init__(self, vehicle_ids: list[str]) -> None:
        self.vehicles = {
            vehicle_id: SimpleNamespace(name="Car") for vehicle_id in vehicle_ids
        }
        self.force_failures: set[str] = set()
        self.cached_failures: set[str] = set()
        self.calls: list[tuple[str, str]] = []

    def check_and_force_update_vehicle(self, interval: int, vehicle_id: str) -> None:
        self.calls.append(("force", vehicle_id))
        if vehicle_id in self.force_failures:
            raise TimeoutError

    def update_vehicle_with_cached_state(self, vehicle_id: str) -> None:
        self.calls.append(("cached", vehicle_id))
        if vehicle_id in self.cached_failures:
            raise TimeoutError


def make_coordinator(*vehicle_ids: str) -> Any:
    """A coordinator polling in force refresh hours, bypassing HA setup."""
    coordinator: Any = object.__new__(HyundaiKiaConnectDataUpdateCoordinator)
    coordinator.force_refresh_interval = 24 * 60 * 60
    coordinator._failed_vehicles = set()
    coordinator._vehicle_failures = {}
    coordinator._force_refresh_failed = set()
    coordinator.vehicle_manager = FakeManager(list(vehicle_ids or ["a"]))

    class Hass:
        async def async_add_executor_job(self, func: Any, *args: Any) -> Any:
            return func(*args)

    coordinator.hass = Hass()
    return coordinator


async def poll(coordinator: Any) -> list[tuple[str, str]]:
    """Run one scheduled poll and return the calls it made."""
    manager = coordinator.vehicle_manager
    manager.calls = []
    await coordinator._async_poll_vehicles(mode="check")
    return manager.calls


async def test_failed_force_refresh_reads_the_cache_before_forcing_again() -> None:
    coordinator = make_coordinator()
    coordinator.vehicle_manager.force_failures.add("a")

    assert await poll(coordinator) == [("force", "a"), ("cached", "a")]
    assert await poll(coordinator) == [("cached", "a")]
    assert await poll(coordinator) == [("force", "a"), ("cached", "a")]


async def test_force_refresh_waits_for_a_successful_cache_read() -> None:
    coordinator = make_coordinator()
    manager = coordinator.vehicle_manager
    manager.force_failures.add("a")
    await poll(coordinator)

    manager.cached_failures.add("a")
    with pytest.raises(UpdateFailed):
        await poll(coordinator)
    manager.cached_failures.clear()

    assert await poll(coordinator) == [("cached", "a")]


async def test_successful_force_refresh_does_not_hold_back_the_next() -> None:
    coordinator = make_coordinator()

    assert await poll(coordinator) == [("force", "a")]
    assert await poll(coordinator) == [("force", "a")]


async def test_failed_force_refresh_holds_back_only_its_vehicle() -> None:
    coordinator = make_coordinator("a", "b")
    coordinator.vehicle_manager.force_failures.add("a")
    await poll(coordinator)

    assert await poll(coordinator) == [("cached", "a"), ("force", "b")]
