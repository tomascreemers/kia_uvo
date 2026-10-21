"""Lifetime driving energy and distance meters for Home Assistant statistics.

The EU ``drvhistory`` endpoint only returns rolling windows: one row per
driving day for about 30 days, and a 90-day total. Neither can feed Home
Assistant's long-term statistics directly, because a rolling window shrinks
as old days drop out. This module turns the daily rows into meters that only
move when driving is reported (or, rarely, corrected): every poll applies the
change of each day's row since the previous poll.

The car uploads its trips on its own, sometimes days late: a forced refresh
does not make it. A change to today's row moves the meter, so it lands in the
hour it is reported. A change to an earlier day's row is kept apart as
``late`` until the sensor has added it to that day's statistics, so the
energy and distance show up on the day they were driven.

Rows are forgotten by age, never because a response lacked them. The library
reports no rows when the endpoint returns nothing usable, and forgetting rows
then would count all 30 days again once they come back.

The rows seen when a meter starts are not counted by the meter itself;
``backfill_statistics`` turns them into hourly statistics placed before the
meter's first state, so the ~30 days before installation still show up.

Home Assistant computes the change of the first statistics row against 0,
so the imported sums start at 0 and end at the imported total. The recorder
continues the sum from the sensor's last short-term row, so the sensor shifts
the rows compiled since the meter started up by that total.

Each daily row also splits its energy into the motor, climate, electronics
and battery care parts, which add up to the row's total. Each part is a meter
of its own, kept exactly like the total. The motor part is net of the
regenerated energy, which the row reports separately.

Driving reported more than ~30 days late never shows up as a daily row, so
the meters cannot count it. It is only detected: the API's rolling 90-day
total then grows by more than the daily rows, and the tracker logs a
warning.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from typing import Any, Literal, Protocol

from hyundai_kia_connect_api import Vehicle
from hyundai_kia_connect_api.const import LENGTH_MILES

_LOGGER = logging.getLogger(__name__)

# Longer than the ~30 days the API returns, so a row is never forgotten
# while the API can still revise it.
ROW_RETENTION = timedelta(days=35)
KM_PER_MILE = 1.609344
# Days before the meter: the hour backfill_statistics put them in.
BACKFILL_HOUR = time(12)
# Days since the meter: the API gives no time, and a late change is driving
# after the last report of that day, so book it in the day's last hour.
LATE_HOUR = time(23)

Quantity = Literal[
    "energy",
    "distance",
    "energy_motor",
    "energy_climate",
    "energy_electronics",
    "energy_battery_care",
]
EnergyPart = Literal[
    "energy_motor",
    "energy_climate",
    "energy_electronics",
    "energy_battery_care",
]
# Each part's ``DailyDrivingStats`` attribute. They add up to
# ``total_consumed``.
ENERGY_PARTS: dict[EnergyPart, str] = {
    "energy_motor": "engine_consumption",
    "energy_climate": "climate_consumption",
    "energy_electronics": "onboard_electronics_consumption",
    "energy_battery_care": "battery_care_consumption",
}
QUANTITY_UNITS: dict[Quantity, str] = {
    "energy": "Wh",
    "distance": "km",
    **{part: "Wh" for part in ENERGY_PARTS},
}


@dataclass
class DailyTotal:
    """A total of one quantity, built from the change of its daily rows.

    ``total`` counts changes to the current day's row; it is the sensor state.
    ``late`` holds changes to earlier days until they are added to those
    days' statistics, after which they count in ``backdated``.
    """

    total: float = 0.0
    rows: dict[str, float] = field(default_factory=dict)
    started: str | None = None
    started_at: str | None = None
    baseline_rows: dict[str, float] = field(default_factory=dict)
    backfilled: bool = False
    late: dict[str, float] = field(default_factory=dict)
    backdated: float = 0.0

    @property
    def initialized(self) -> bool:
        """Return True once a first set of daily rows has been seen."""
        return self.started is not None

    @property
    def measured(self) -> float:
        """Return everything counted since the start, wherever it is booked."""
        return self.total + self.backdated + sum(self.late.values())

    def update(
        self,
        rows: Mapping[date, float] | None,
        now: datetime,
    ) -> bool:
        """Apply the latest daily rows; return True if anything changed.

        The first non-empty set of rows only sets the baseline: driving
        before the meter existed is not booked as new.
        """
        if not rows:
            # No information: keep everything as it is.
            return False
        today = now.date()
        cutoff = (today - ROW_RETENTION).isoformat()
        latest = {
            day.isoformat(): float(value)
            for day, value in rows.items()
            if day.isoformat() >= cutoff
        }
        changed = False
        if not self.initialized:
            self.rows = latest
            self.baseline_rows = dict(latest)
            self.started = today.isoformat()
            self.started_at = now.isoformat()
            changed = True
        else:
            today_key = today.isoformat()
            for key, value in latest.items():
                delta = value - self.rows.get(key, 0.0)
                if delta or key not in self.rows:
                    if key < today_key:
                        self.late[key] = self.late.get(key, 0.0) + delta
                    else:
                        self.total += delta
                    self.rows[key] = value
                    changed = True
        kept = {key: value for key, value in self.rows.items() if key >= cutoff}
        if len(kept) != len(self.rows):
            self.rows = kept
            changed = True
        return changed

    def mark_backdated(self, applied: Mapping[str, float]) -> None:
        """Record that ``applied`` late changes were added to statistics."""
        for key, value in applied.items():
            remaining = self.late.get(key, 0.0) - value
            if abs(remaining) < 1e-9:
                self.late.pop(key, None)
            else:
                self.late[key] = remaining
            self.backdated += value

    def late_hour(self, key: str, tz: tzinfo) -> datetime:
        """Return the statistics hour a late change to day ``key`` belongs in."""
        day = date.fromisoformat(key)
        if self.started is not None and key < self.started:
            return datetime.combine(day, BACKFILL_HOUR, tz)
        return datetime.combine(day, LATE_HOUR, tz)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DailyTotal:
        """Build a total from ``asdict`` output."""
        return cls(
            total=float(data.get("total", 0.0)),
            rows=_float_rows(data.get("rows", {})),
            started=data.get("started"),
            started_at=data.get("started_at"),
            baseline_rows=_float_rows(data.get("baseline_rows", {})),
            backfilled=bool(data.get("backfilled", False)),
            late=_float_rows(data.get("late", {})),
            backdated=float(data.get("backdated", 0.0)),
        )


def _float_rows(rows: Mapping[Any, Any]) -> dict[str, float]:
    return {str(k): float(v) for k, v in rows.items()}


@dataclass
class DrivingMeter:
    """Driving energy (Wh), its parts (Wh) and distance (km) of one vehicle."""

    energy: DailyTotal = field(default_factory=DailyTotal)
    distance: DailyTotal = field(default_factory=DailyTotal)
    energy_motor: DailyTotal = field(default_factory=DailyTotal)
    energy_climate: DailyTotal = field(default_factory=DailyTotal)
    energy_electronics: DailyTotal = field(default_factory=DailyTotal)
    energy_battery_care: DailyTotal = field(default_factory=DailyTotal)
    # The API's energy total over a rolling 90-day window. Its per-month
    # breakdown (not exposed by the library) shows the oldest month shrinking
    # day by day, so it is not whole calendar months.
    window_wh: float | None = None
    window_carry_wh: float = 0.0
    # Growth of that total not explained by the daily rows: driving reported
    # for days older than the daily window, which the meters cannot count.
    missed_wh: float = 0.0

    @property
    def initialized(self) -> bool:
        """Return True once the energy total has started."""
        return self.energy.initialized

    def update(
        self,
        energy_rows: Mapping[date, float] | None,
        distance_rows: Mapping[date, float] | None,
        now: datetime,
        window_wh: float | None = None,
    ) -> bool:
        """Apply the latest daily rows; return True if anything changed."""
        measured = self.energy.measured
        changed = self.energy.update(energy_rows, now)
        if energy_rows and window_wh is not None:
            # Only with daily rows: otherwise their growth is unknown.
            changed = (
                self._check_window(float(window_wh), self.energy.measured - measured)
                or changed
            )
        # Not short-circuited: both totals must see every poll.
        return self.distance.update(distance_rows, now) or changed

    def update_parts(
        self,
        part_rows: Mapping[EnergyPart, Mapping[date, float] | None],
        now: datetime,
    ) -> bool:
        """Apply the latest daily rows of the energy parts."""
        changed = False
        for part, rows in part_rows.items():
            total: DailyTotal = getattr(self, part)
            # Not short-circuited: every part must see every poll.
            changed = total.update(rows, now) or changed
        return changed

    def _check_window(self, window_wh: float, counted_wh: float) -> bool:
        """Add growth of the 90-day total beyond ``counted_wh`` to ``missed_wh``.

        The two totals come from separate requests, so a trip reported in
        between shows as a shortfall in one poll and an excess in the next;
        a shortfall is carried to the next poll once. The day dropping out of
        the window each day is a shortfall too, and is carried once as well,
        so a late trip smaller than that day's driving, arriving with the
        drop or in the next poll, goes unnoticed.
        """
        if self.window_wh is None:
            self.window_wh = window_wh
            return True
        if window_wh == self.window_wh and not counted_wh and not self.window_carry_wh:
            return False
        raw = window_wh - self.window_wh - counted_wh
        unexplained = raw + self.window_carry_wh
        self.window_wh = window_wh
        self.window_carry_wh = raw if raw < 0 else 0.0
        if unexplained > 0:
            self.missed_wh += unexplained
        return True

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DrivingMeter:
        """Build a meter from ``as_dict`` output."""
        return cls(
            energy=DailyTotal.from_dict(data.get("energy", {})),
            distance=DailyTotal.from_dict(data.get("distance", {})),
            **{part: DailyTotal.from_dict(data.get(part, {})) for part in ENERGY_PARTS},
            window_wh=data.get("window_wh"),
            window_carry_wh=float(data.get("window_carry_wh", 0.0)),
            missed_wh=float(data.get("missed_wh", 0.0)),
        )


@dataclass
class Backfill:
    """Hourly statistics for the days before a meter started."""

    # (hour start, sum) rows; sums run from the first day's value up to total.
    rows: list[tuple[datetime, float]]
    total: float
    # The first hour the recorder compiles for the meter.
    meter_start: datetime


def backfill_statistics(total: DailyTotal, tz: tzinfo) -> Backfill | None:
    """Return statistics for the days before the meter, or None if unstarted.

    The API only gives per-day totals, so each day is placed at 12:00 local
    time. Every row lies strictly before the meter's first state: importing
    over hours the recorder already compiled would break the sum chain.
    """
    if total.started_at is None:
        return None
    days = total.baseline_rows
    started_at = datetime.fromisoformat(total.started_at).astimezone(tz)
    meter_start = started_at.replace(minute=0, second=0, microsecond=0)
    latest_hour = meter_start - timedelta(hours=1)
    placed: list[tuple[datetime, float]] = []
    for key in sorted(days):
        # The start day's row can fall after the meter's first state.
        start = min(
            datetime.combine(date.fromisoformat(key), BACKFILL_HOUR, tz), latest_hour
        )
        if placed and start <= placed[-1][0]:
            # Two days in the same hour: merge into the earlier row.
            placed[-1] = (placed[-1][0], placed[-1][1] + days[key])
            continue
        placed.append((start, days[key]))
    rows: list[tuple[datetime, float]] = []
    running = 0.0
    for start, value in placed:
        running += value
        rows.append((start, running))
    return Backfill(rows, running, meter_start)


def daily_rows(
    vehicle: Vehicle, attribute: str = "total_consumed"
) -> dict[date, float] | None:
    """Return the vehicle's daily rows (Wh) of an energy attribute by date."""
    stats = vehicle.daily_stats
    if not stats:
        return None
    rows: dict[date, float] = {}
    for day in stats:
        value = getattr(day, attribute)
        if day.date is None or value is None:
            continue
        rows[day.date.date()] = float(value)
    return rows or None


def daily_part_rows(
    vehicle: Vehicle,
) -> dict[EnergyPart, dict[date, float] | None]:
    """Return the vehicle's daily rows (Wh) of each energy part."""
    return {
        part: daily_rows(vehicle, attribute) for part, attribute in ENERGY_PARTS.items()
    }


def daily_distance_rows(vehicle: Vehicle) -> dict[date, float] | None:
    """Return the vehicle's daily driving distance rows (km) keyed by date."""
    stats = vehicle.daily_stats
    if not stats:
        return None
    rows: dict[date, float] = {}
    for day in stats:
        if day.date is None or day.distance is None:
            continue
        km = float(day.distance)
        if day.distance_unit == LENGTH_MILES:
            km *= KM_PER_MILE
        rows[day.date.date()] = km
    return rows or None


class MeterStore(Protocol):
    """The subset of ``homeassistant.helpers.storage.Store`` used here."""

    async def async_load(self) -> dict[str, Any] | None:
        """Load stored data."""

    async def async_save(self, data: dict[str, Any]) -> None:
        """Save data."""


class DrivingMeterTracker:
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
        self.meter: DrivingMeter | None = None

    async def async_load(self) -> None:
        """Load the persisted meter."""
        data = await self._store.async_load()
        self.meter = DrivingMeter.from_dict(data) if data else DrivingMeter()
        if not data:
            _LOGGER.debug("%s: no stored driving meter yet", self._vehicle.name)
            return
        for quantity, unit in QUANTITY_UNITS.items():
            total: DailyTotal = getattr(self.meter, quantity)
            _LOGGER.debug(
                "%s: loaded driving %s meter: total %s %s since %s, %d days of "
                "rows, backfilled %s, late %s, booked late so far %s",
                self._vehicle.name,
                quantity,
                total.total,
                unit,
                total.started_at,
                len(total.rows),
                total.backfilled,
                total.late,
                total.backdated,
            )

    async def async_process(self) -> None:
        """Apply the vehicle's current rows; save before publishing."""
        now = self._now()
        before = self.meter
        missed = before.missed_wh if before is not None else 0.0
        energy_rows = daily_rows(self._vehicle)
        part_rows = daily_part_rows(self._vehicle)

        def update(meter: DrivingMeter) -> bool:
            changed = meter.update(
                energy_rows,
                daily_distance_rows(self._vehicle),
                now,
                self._vehicle.total_power_consumed,
            )
            # Not short-circuited: the parts must see every poll.
            return meter.update_parts(part_rows, now) or changed

        if not await self._async_change(update):
            _LOGGER.debug(
                "%s: driving meters unchanged (%d daily rows)",
                self._vehicle.name,
                len(energy_rows or {}),
            )
            return
        if before is not None and self.meter is not None:
            self._log_changes(before, self.meter)
        if self.meter is not None and self.meter.missed_wh > missed:
            _LOGGER.warning(
                "%s: %.0f Wh of driving was reported for days older than the "
                "~30 days of daily rows. It is not in Driving Energy or Driving "
                "Distance (%.0f Wh missed so far)",
                self._vehicle.name,
                self.meter.missed_wh - missed,
                self.meter.missed_wh,
            )
        for listener in list(self._listeners):
            listener()

    def _log_changes(self, before: DrivingMeter, after: DrivingMeter) -> None:
        """Log what one poll changed in each total."""
        for quantity, unit in QUANTITY_UNITS.items():
            old: DailyTotal = getattr(before, quantity)
            new: DailyTotal = getattr(after, quantity)
            if not old.initialized and new.initialized:
                _LOGGER.debug(
                    "%s: driving %s meter started at %s, %d days before it "
                    "(%s %s) left for the backfill",
                    self._vehicle.name,
                    quantity,
                    new.started_at,
                    len(new.baseline_rows),
                    sum(new.baseline_rows.values()),
                    unit,
                )
                continue
            changes = {
                key: value - old.rows.get(key, 0.0)
                for key, value in sorted(new.rows.items())
                if key not in old.rows or value != old.rows[key]
            }
            dropped = sorted(old.rows.keys() - new.rows.keys())
            if not changes and not dropped:
                continue
            _LOGGER.debug(
                "%s: driving %s rows changed by %s %s, dropped %s; total %s -> %s, "
                "late %s",
                self._vehicle.name,
                quantity,
                changes,
                unit,
                dropped,
                old.total,
                new.total,
                new.late,
            )
        if after.window_wh != before.window_wh:
            _LOGGER.debug(
                "%s: 90-day driving energy total %s -> %s Wh, carried %s Wh",
                self._vehicle.name,
                before.window_wh,
                after.window_wh,
                after.window_carry_wh,
            )

    async def async_mark_backfilled(self, quantity: Quantity) -> None:
        """Record that the days before ``quantity``'s start were imported."""

        def mark(meter: DrivingMeter) -> bool:
            getattr(meter, quantity).backfilled = True
            return True

        await self._async_change(mark)

    async def async_mark_backdated(
        self, quantity: Quantity, applied: Mapping[str, float]
    ) -> None:
        """Record that ``applied`` late changes were added to statistics."""

        def mark(meter: DrivingMeter) -> bool:
            getattr(meter, quantity).mark_backdated(applied)
            return True

        await self._async_change(mark)

    async def _async_change(self, change: Callable[[DrivingMeter], bool]) -> bool:
        """Apply ``change`` to a copy; save, then keep it if it changed."""
        async with self._lock:
            if self.meter is None:
                return False
            updated = copy.deepcopy(self.meter)
            if not change(updated):
                return False
            # Persist first: a crash after publishing but before saving would
            # restore a lower total, which statistics book as negative usage.
            await self._store.async_save(updated.as_dict())
            self.meter = updated
            return True

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` after each published change; return a remover."""
        self._listeners.append(listener)

        def remove() -> None:
            self._listeners.remove(listener)

        return remove
