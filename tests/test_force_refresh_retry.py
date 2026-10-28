"""Tests that a failed force refresh is not retried before reading the cache."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.kia_uvo.coordinator import HyundaiKiaConnectDataUpdateCoordinator


def make_coordinator() -> Any:
    """A coordinator allowed to force refresh at any hour, bypassing HA setup."""
    coordinator: Any = object.__new__(HyundaiKiaConnectDataUpdateCoordinator)
    coordinator.scan_interval = 30 * 60
    coordinator.force_refresh_interval = 24 * 60 * 60
    coordinator.no_force_refresh_hour_start = 0
    coordinator.no_force_refresh_hour_finish = 0
    coordinator._force_refresh_failed = False
    coordinator.data = {}
    coordinator.async_check_and_refresh_token = AsyncMock()
    coordinator.vehicle_manager = MagicMock()

    async def run(func: Any, *args: Any) -> Any:
        return func(*args)

    coordinator.hass = MagicMock()
    coordinator.hass.async_add_executor_job = run
    return coordinator


def calls(coordinator: Any) -> tuple[int, int]:
    """Return how often the polls forced a refresh and read the cache."""
    manager = coordinator.vehicle_manager
    return (
        manager.check_and_force_update_vehicles.call_count,
        manager.update_all_vehicles_with_cached_state.call_count,
    )


async def test_failed_force_refresh_reads_the_cache_before_forcing_again() -> None:
    coordinator = make_coordinator()
    coordinator.vehicle_manager.check_and_force_update_vehicles.side_effect = (
        TimeoutError
    )

    await coordinator._async_update_data()
    assert calls(coordinator) == (1, 1)

    await coordinator._async_update_data()
    assert calls(coordinator) == (1, 2)

    await coordinator._async_update_data()
    assert calls(coordinator) == (2, 3)


async def test_force_refresh_waits_for_a_successful_cache_read() -> None:
    coordinator = make_coordinator()
    manager = coordinator.vehicle_manager
    manager.check_and_force_update_vehicles.side_effect = TimeoutError
    await coordinator._async_update_data()

    manager.update_all_vehicles_with_cached_state.side_effect = TimeoutError
    with pytest.raises(TimeoutError):
        await coordinator._async_update_data()
    manager.update_all_vehicles_with_cached_state.side_effect = None

    await coordinator._async_update_data()
    assert calls(coordinator) == (1, 3)


async def test_successful_force_refresh_does_not_hold_back_the_next() -> None:
    coordinator = make_coordinator()

    await coordinator._async_update_data()
    await coordinator._async_update_data()
    assert calls(coordinator) == (2, 0)
