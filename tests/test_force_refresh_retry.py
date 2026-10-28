"""Tests that a failed force refresh is not retried on every poll."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from custom_components.kia_uvo import coordinator as coordinator_module
from custom_components.kia_uvo.coordinator import HyundaiKiaConnectDataUpdateCoordinator

START = datetime(2026, 10, 9, 19, 18, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    clock = Clock()
    monkeypatch.setattr(coordinator_module.dt_util, "utcnow", lambda: clock.now)
    monkeypatch.setattr(coordinator_module.dt_util, "now", lambda: clock.now)
    return clock


def make_coordinator(force_refresh_minutes: int) -> Any:
    """A coordinator with the polling settings, bypassing HA setup."""
    coordinator: Any = object.__new__(HyundaiKiaConnectDataUpdateCoordinator)
    coordinator.scan_interval = 30 * 60
    coordinator.force_refresh_interval = force_refresh_minutes * 60
    # Force refresh allowed all day.
    coordinator.no_force_refresh_hour_start = 0
    coordinator.no_force_refresh_hour_finish = 0
    coordinator._force_refresh_retry_at = None
    coordinator.data = {}
    coordinator.async_check_and_refresh_token = AsyncMock()
    coordinator.vehicle_manager = MagicMock()

    async def run(func: Any, *args: Any) -> Any:
        return func(*args)

    coordinator.hass = MagicMock()
    coordinator.hass.async_add_executor_job = run
    return coordinator


def forced(coordinator: Any) -> int:
    return int(coordinator.vehicle_manager.check_and_force_update_vehicles.call_count)


async def test_failed_force_refresh_waits_before_waking_the_car_again(
    clock: Clock,
) -> None:
    coordinator = make_coordinator(force_refresh_minutes=24 * 60)
    manager = coordinator.vehicle_manager
    manager.check_and_force_update_vehicles.side_effect = TimeoutError

    await coordinator._async_update_data()
    assert forced(coordinator) == 1
    assert manager.update_all_vehicles_with_cached_state.call_count == 1

    # The next polls only read the cache.
    for _ in range(11):
        clock.advance(minutes=30)
        await coordinator._async_update_data()
    assert forced(coordinator) == 1
    assert manager.update_all_vehicles_with_cached_state.call_count == 12

    clock.advance(minutes=30)
    manager.check_and_force_update_vehicles.side_effect = None
    await coordinator._async_update_data()
    assert forced(coordinator) == 2


async def test_retry_delay_never_exceeds_the_force_refresh_interval(
    clock: Clock,
) -> None:
    coordinator = make_coordinator(force_refresh_minutes=60)
    coordinator.vehicle_manager.check_and_force_update_vehicles.side_effect = (
        TimeoutError
    )
    await coordinator._async_update_data()
    clock.advance(minutes=30)
    await coordinator._async_update_data()
    assert forced(coordinator) == 1
    clock.advance(minutes=30)
    await coordinator._async_update_data()
    assert forced(coordinator) == 2
