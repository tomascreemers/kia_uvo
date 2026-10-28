"""Tests for the dev-only force refresh probe."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from custom_components.kia_uvo import force_refresh_probe

BASE = "https://example.test/api/v1/spa/"


def make_manager(response: Any) -> Any:
    session = MagicMock(HTTP_CONNECT_TIMEOUT=10)
    session.get.return_value = response
    manager = MagicMock()
    manager.api.SPA_API_URL = BASE
    manager.api.session = session
    return manager


def response(body: Any) -> Any:
    return MagicMock(status_code=200, json=MagicMock(return_value=body))


def test_forced_status_waits_longer_and_is_recorded() -> None:
    hass = MagicMock()
    manager = make_manager(
        response({"retCode": "F", "resCode": "4081", "resMsg": "Request timeout"})
    )
    get = manager.api.session.get
    force_refresh_probe.install(hass, manager)

    manager.api.session.get(BASE + "vehicles/v1/status", headers={})

    assert get.call_args.kwargs["timeout"] == (10, force_refresh_probe.READ_TIMEOUT)
    entity_id, state, attributes = hass.states.set.call_args.args
    assert entity_id == force_refresh_probe.ENTITY_ID
    assert state.startswith("F 4081 Request timeout after ")
    assert "started_at" in attributes


@pytest.mark.parametrize("path", ["vehicles/v1/status/latest", "vehicles/v1/location"])
def test_other_requests_are_left_alone(path: str) -> None:
    hass = MagicMock()
    manager = make_manager(response({"retCode": "S", "resCode": "0000"}))
    get = manager.api.session.get
    force_refresh_probe.install(hass, manager)

    manager.api.session.get(BASE + path, headers={})

    assert "timeout" not in get.call_args.kwargs
    hass.states.set.assert_not_called()


def test_timeouts_are_recorded_and_raised() -> None:
    hass = MagicMock()
    manager = make_manager(None)
    manager.api.session.get.side_effect = TimeoutError
    force_refresh_probe.install(hass, manager)

    with pytest.raises(TimeoutError):
        manager.api.session.get(BASE + "vehicles/v1/status", headers={})

    assert hass.states.set.call_args.args[1].startswith("TimeoutError after ")
