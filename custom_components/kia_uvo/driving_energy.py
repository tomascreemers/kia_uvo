"""Lifetime driving energy meter for Home Assistant statistics.

The EU ``drvhistory`` endpoint only returns rolling windows: one row per
driving day for about 30 days, and a 90-day total. Neither can feed Home
Assistant's long-term statistics directly, because a rolling window shrinks
as old days drop out. This module turns the daily rows into a meter that
only moves when driving energy is added (or, rarely, corrected): every poll
adds the change of each day's row since the previous poll.

Rows are forgotten by age, never because a response lacked them. The library
reports no rows when the endpoint returns nothing usable, and forgetting rows
then would count all 30 days again once they come back.

The rows seen when the meter starts are not counted by the meter itself;
``backfill_statistics`` turns them into hourly statistics placed before the
meter's first state, so the ~30 days before installation still show up.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any, Protocol

from hyundai_kia_connect_api import Vehicle
from hyundai_kia_connect_api.const import LENGTH_MILES

# Longer than the ~30 days the API returns, so a row is never forgotten
# while the API can still revise it.
ROW_RETENTION = timedelta(days=35)
KM_PER_MILE = 1.609344


@dataclass
class DrivingEnergyMeter:
    """Accumulated driving energy (Wh) and the daily rows it was built from."""

    total_wh: float = 0.0
    rows: dict[str, float] = field(default_factory=dict)
    started: str | None = None
    odometer_at_start_km: float | None = None
    api_wh_per_km_at_start: float | None = None
    started_at: str | None = None
    baseline_rows: dict[str, float] | None = None
    backfilled: bool = False

    @property
    def initialized(self) -> bool:
        """Return True once a first set of daily rows has been seen."""
        return self.started is not None

    def update(
        self,
        rows: Mapping[date, float] | None,
        today: date,
        odometer_km: float | None = None,
        api_wh_per_km: float | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Apply the latest daily rows; return True if anything changed.

        The first non-empty set of rows only sets the baseline: energy driven
        before the meter existed is not booked as new consumption.
        """
        if not rows:
            # No information: keep everything as it is.
            return False
        cutoff = (today - ROW_RETENTION).isoformat()
        latest = {
            day.isoformat(): float(wh)
            for day, wh in rows.items()
            if day.isoformat() >= cutoff
        }
        changed = False
        if not self.initialized:
            self.rows = latest
            self.baseline_rows = dict(latest)
            self.started = today.isoformat()
            self.started_at = now.isoformat() if now is not None else None
            self.odometer_at_start_km = odometer_km
            self.api_wh_per_km_at_start = api_wh_per_km
            changed = True
        else:
            for key, wh in latest.items():
                delta = wh - self.rows.get(key, 0.0)
                if delta or key not in self.rows:
                    self.total_wh += delta
                    self.rows[key] = wh
                    changed = True
        kept = {key: wh for key, wh in self.rows.items() if key >= cutoff}
        if len(kept) != len(self.rows):
            self.rows = kept
            changed = True
        return changed

    def lifetime_estimate_wh(self, wh_per_km: float | None) -> float | None:
        """Estimate lifetime driving energy: start odometer x average + measured."""
        if wh_per_km is None:
            wh_per_km = self.api_wh_per_km_at_start
        if (
            not self.initialized
            or self.odometer_at_start_km is None
            or wh_per_km is None
        ):
            return None
        return self.odometer_at_start_km * wh_per_km + self.total_wh

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DrivingEnergyMeter:
        """Build a meter from ``as_dict`` output."""
        return cls(
            total_wh=float(data.get("total_wh", 0.0)),
            rows={str(k): float(v) for k, v in data.get("rows", {}).items()},
            started=data.get("started"),
            odometer_at_start_km=data.get("odometer_at_start_km"),
            api_wh_per_km_at_start=data.get("api_wh_per_km_at_start"),
            started_at=data.get("started_at"),
            baseline_rows=(
                {str(k): float(v) for k, v in baseline.items()}
                if (baseline := data.get("baseline_rows")) is not None
                else None
            ),
            backfilled=bool(data.get("backfilled", False)),
        )


def backfill_statistics(
    meter: DrivingEnergyMeter, tz: tzinfo
) -> list[tuple[datetime, float]]:
    """Return ``(hour start, sum in Wh)`` rows for the days before the meter.

    The API only gives per-day totals, so each day is placed at 12:00 local
    time. Sums are rebased to end at 0, the meter's zero point in statistics,
    and every row lies strictly before the meter's first state: importing
    over hours the recorder already compiled would break the sum chain.
    """
    if not meter.initialized or meter.started is None:
        return []
    if meter.baseline_rows is not None:
        days = dict(meter.baseline_rows)
    else:
        # Meters started before baseline rows were stored: only days before
        # the start date are certainly free of energy the meter counted.
        days = {k: v for k, v in meter.rows.items() if k < meter.started}
    latest_hour: datetime | None = None
    if meter.started_at is not None:
        started_at = datetime.fromisoformat(meter.started_at).astimezone(tz)
        latest_hour = started_at.replace(minute=0, second=0, microsecond=0) - (
            timedelta(hours=1)
        )
    placed: list[tuple[datetime, float]] = []
    for key in sorted(days):
        start = datetime.combine(date.fromisoformat(key), time(12), tz)
        if latest_hour is not None and start > latest_hour:
            start = latest_hour
        elif latest_hour is None and key >= meter.started:
            continue
        if placed and start <= placed[-1][0]:
            # Two days in the same hour: merge into the earlier row.
            placed[-1] = (placed[-1][0], placed[-1][1] + days[key])
            continue
        placed.append((start, days[key]))
    total = sum(wh for _, wh in placed)
    result: list[tuple[datetime, float]] = []
    running = 0.0
    for start, wh in placed:
        running += wh
        result.append((start, running - total))
    return result


def daily_rows(vehicle: Vehicle) -> dict[date, float] | None:
    """Return the vehicle's daily driving energy rows keyed by date."""
    stats = vehicle.daily_stats
    if not stats:
        return None
    rows: dict[date, float] = {}
    for day in stats:
        if day.date is None or day.total_consumed is None:
            continue
        rows[day.date.date()] = float(day.total_consumed)
    return rows or None


def odometer_km(vehicle: Vehicle) -> float | None:
    """Return the odometer in km, or None when unknown."""
    value = vehicle.odometer
    if value is None:
        return None
    if vehicle.odometer_unit == LENGTH_MILES:
        return float(value) * KM_PER_MILE
    return float(value)


class MeterStore(Protocol):
    """The subset of ``homeassistant.helpers.storage.Store`` used here."""

    async def async_load(self) -> dict[str, Any] | None:
        """Load stored data."""

    async def async_save(self, data: dict[str, Any]) -> None:
        """Save data."""


class DrivingEnergyTracker:
    """Keep one vehicle's meter in sync with coordinator updates."""

    def __init__(
        self,
        store: MeterStore,
        vehicle: Vehicle,
        now: Callable[[], datetime],
    ) -> None:
        """Initialize the tracker."""
        self._store = store
        self._vehicle = vehicle
        self._now = now
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[], None]] = []
        self.meter: DrivingEnergyMeter | None = None

    async def async_load(self) -> None:
        """Load the persisted meter."""
        data = await self._store.async_load()
        self.meter = (
            DrivingEnergyMeter.from_dict(data) if data else DrivingEnergyMeter()
        )

    async def async_process(self) -> None:
        """Apply the vehicle's current rows; save before publishing."""
        async with self._lock:
            if self.meter is None:
                return
            updated = copy.deepcopy(self.meter)
            now = self._now()
            if not updated.update(
                daily_rows(self._vehicle),
                now.date(),
                odometer_km(self._vehicle),
                self._vehicle.power_consumption_30d,
                now,
            ):
                return
            # Persist first: a crash after publishing but before saving would
            # restore a lower total, which statistics book as negative usage.
            await self._store.async_save(updated.as_dict())
            self.meter = updated
        for listener in list(self._listeners):
            listener()

    async def async_mark_backfilled(self) -> None:
        """Record that the pre-meter days were imported into statistics."""
        async with self._lock:
            if self.meter is None:
                return
            updated = copy.deepcopy(self.meter)
            updated.backfilled = True
            await self._store.async_save(updated.as_dict())
            self.meter = updated

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` after each published change; return a remover."""
        self._listeners.append(listener)

        def remove() -> None:
            self._listeners.remove(listener)

        return remove
