"""Experiment for the dev branch only, not for upstream.

Lets the EU (non-CCS2) forced status request wait up to READ_TIMEOUT
instead of the library's 30 s, to learn how long the server holds it and
what it answers when the car does not respond. Every attempt is logged and
recorded in the state of ENTITY_ID, so its history can be read later.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
import logging
import re
import time
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from hyundai_kia_connect_api import VehicleManager
from hyundai_kia_connect_api.KiaUvoApiEU import KiaUvoApiEU

_LOGGER = logging.getLogger(__name__)

READ_TIMEOUT = 300
ENTITY_ID = "sensor.kia_uvo_force_refresh_probe"


def install(hass: HomeAssistant, manager: VehicleManager) -> None:
    """Wrap the API session's GET for the forced status request."""
    api = manager.api
    # AU, IN and CN share the URL, but the experiment is about the EU server.
    if not isinstance(api, KiaUvoApiEU):
        return
    # Any: mypy does not allow replacing the method.
    session: Any = api.session
    forced_status = re.compile(re.escape(api.SPA_API_URL) + r"vehicles/[^/]+/status")
    get: Callable[..., Any] = session.get

    def probed_get(url: str, *args: Any, **kwargs: Any) -> Any:
        if not forced_status.fullmatch(url):
            return get(url, *args, **kwargs)
        kwargs.setdefault("timeout", (session.HTTP_CONNECT_TIMEOUT, READ_TIMEOUT))
        started_at = dt_util.now()
        start = time.monotonic()
        try:
            response = get(url, *args, **kwargs)
        except Exception as err:
            _record(hass, started_at, time.monotonic() - start, type(err).__name__)
            raise
        _record(hass, started_at, time.monotonic() - start, _outcome(response))
        return response

    session.get = probed_get


def _outcome(response: Any) -> str:
    """Return the server's result codes, and its message unless it succeeded."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if not isinstance(body, dict):
        return f"HTTP {response.status_code}"
    codes = f"{body.get('retCode')} {body.get('resCode')}"
    if body.get("retCode") == "S":
        return codes
    return f"{codes} {str(body.get('resMsg'))[:100]}"


def _record(
    hass: HomeAssistant, started_at: datetime, elapsed: float, outcome: str
) -> None:
    try:
        _LOGGER.warning(
            "Forced status request from %s ended after %.0f s: %s",
            started_at.isoformat(timespec="seconds"),
            elapsed,
            outcome,
        )
        hass.states.set(
            ENTITY_ID,
            f"{outcome} after {elapsed:.0f} s"[:255],
            {"started_at": started_at.isoformat(), "elapsed_s": round(elapsed, 1)},
        )
    except Exception:
        _LOGGER.exception("Could not record the forced status request")
