"""Tests for the dev-only force refresh probe."""

from typing import Any
from unittest.mock import MagicMock, patch

from hyundai_kia_connect_api.exceptions import RequestTimeoutError
from hyundai_kia_connect_api.KiaUvoApiEU import KiaUvoApiEU
from hyundai_kia_connect_api.VehicleManager import VehicleManager
import pytest
import requests
from requests.adapters import HTTPAdapter

from custom_components.kia_uvo import force_refresh_probe

BASE = "https://example.test/api/v1/spa/"


def make_manager(response: Any) -> Any:
    session = MagicMock(HTTP_CONNECT_TIMEOUT=10)
    session.get.return_value = response
    manager = MagicMock()
    manager.api = MagicMock(KiaUvoApiEU)
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


def library_api(region: int) -> Any:
    """The library's API for ``region`` (1 EU, 5 AU), Kia, with stub headers."""
    api = VehicleManager.get_implementation_by_region_brand(region, 1, "en")
    api._get_vehicle_headers = lambda token, vehicle: {}
    return api


def forced_status(api: Any, send: Any) -> None:
    vehicle = MagicMock(id="v1", ccu_ccs2_protocol_support=0)
    with patch.object(HTTPAdapter, "send", send):
        api._get_forced_vehicle_state(MagicMock(), vehicle)


def test_library_forced_status_waits_longer_and_records_its_timeout() -> None:
    hass = MagicMock()
    api = library_api(1)
    force_refresh_probe.install(hass, MagicMock(api=api))
    timeouts = []

    def send(adapter: Any, request: Any, **kwargs: Any) -> Any:
        timeouts.append(kwargs["timeout"])
        raise requests.exceptions.ReadTimeout("read timed out")

    with pytest.raises(RequestTimeoutError):
        forced_status(api, send)

    assert timeouts == [(10, force_refresh_probe.READ_TIMEOUT)]
    assert hass.states.set.call_args.args[1].startswith("RequestTimeoutError after ")


def test_other_regions_are_left_alone() -> None:
    hass = MagicMock()
    api = library_api(5)
    get = api.session.get
    force_refresh_probe.install(hass, MagicMock(api=api))
    assert api.session.get == get
