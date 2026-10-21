"""Driving energy and distance meters that are safe for long-term statistics.

The EU ``drvhistory`` endpoint only reports rolling windows: one row per
driving day for about 30 days, and a 90-day energy total. Neither can feed
Home Assistant's long-term statistics directly, because a rolling window
shrinks as old days drop out. These meters instead add up how each daily row
changes from one poll to the next, so they only move when driving is
reported (or, rarely, corrected).

The car uploads its trips on its own, sometimes days late: a forced refresh
does not make it. A change to today's row moves the meter, so it lands in the
hour it is reported. A change to an earlier day's row is kept apart as
``late`` until the sensor has added it to that day's statistics, so the
driving shows up on the day it happened.

Rows are forgotten by age, never because a response lacked them. The library
reports no rows when the endpoint returns nothing usable, and forgetting rows
then would count all 30 days again once they come back.

The rows seen when a meter starts are not counted by the meter itself:
``backfill_statistics`` turns them into statistics placed before the meter's
first state, so the ~30 days before installation still show up.

Each daily row also splits its energy into the motor, climate, electronics
and battery care parts, which add up to the row's total, and reports the
regenerated energy, which the motor part is already net of. Each is a meter
of its own, kept exactly like the total.

Driving reported more than ~30 days late never shows up as a daily row, so
the meters cannot count it. It is only detected: the 90-day total then grows
by more than the daily rows, and the tracker logs a warning.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, tzinfo
from operator import attrgetter
from typing import Any, Literal, Protocol

from hyundai_kia_connect_api import Vehicle
from hyundai_kia_connect_api.const import LENGTH_MILES
from hyundai_kia_connect_api.Vehicle import DailyDrivingStats

_LOGGER = logging.getLogger(__name__)

# Longer than the ~30 days the API returns, so a row is never forgotten
# while the API can still revise it.
ROW_RETENTION = timedelta(days=35)
KM_PER_MILE = 1.609344
# The API gives no time of day. The days before the meter go in this hour.
BACKFILL_HOUR = time(12)
# A late change to a day since the meter started is driving after the last
# report of that day, so it goes in the day's last hour.
LATE_HOUR = time(23)

Quantity = Literal[
    "energy",
    "distance",
    "energy_motor",
    "energy_climate",
    "energy_electronics",
    "energy_battery_care",
    "energy_regenerated",
]
EnergyPart = Literal[
    "energy_motor",
    "energy_climate",
    "energy_electronics",
    "energy_battery_care",
    "energy_regenerated",
]
# Each part's ``DailyDrivingStats`` attribute. All but the regenerated energy
# add up to ``total_consumed``.
ENERGY_PARTS: dict[EnergyPart, str] = {
    "energy_motor": "engine_consumption",
    "energy_climate": "climate_consumption",
    "energy_electronics": "onboard_electronics_consumption",
    "energy_battery_care": "battery_care_consumption",
    "energy_regenerated": "regenerated_energy",
}
QUANTITY_UNITS: dict[Quantity, str] = {
    "energy": "Wh",
    "distance": "km",
    **{part: "Wh" for part in ENERGY_PARTS},
}


@dataclass
class DailyTotal:
    """The total of one quantity, built from the changes of its daily rows.

    ``total`` counts changes to the current day's row; it is the sensor state.
    Changes to earlier days wait in ``late`` until the sensor has added them
    to those days' statistics, and then count in ``backdated``.
    """

    total: float = 0.0
    # The latest value of each day's row.
    rows: dict[date, float] = field(default_factory=dict)
    started_at: datetime | None = None
    # The rows seen at the start, for the backfill.
    baseline_rows: dict[date, float] = field(default_factory=dict)
    backfilled: bool = False
    late: dict[date, float] = field(default_factory=dict)
    backdated: float = 0.0

    @property
    def initialized(self) -> bool:
        """Return True once a first set of daily rows has been seen."""
        return self.started_at is not None

    @property
    def measured(self) -> float:
        """Return everything counted since the start, wherever it is booked."""
        return self.total + self.backdated + sum(self.late.values())

    @property
    def statistics_pending(self) -> bool:
        """Return True while the backfill or late changes await statistics."""
        return self.initialized and (not self.backfilled or bool(self.late))

    def update(self, rows: Mapping[date, float], now: datetime) -> None:
        """Apply the latest daily rows.

        The first rows only set the baseline: driving before the meter
        existed is left to the backfill.
        """
        if not rows:
            # No information: keep everything as it is.
            return
        today = now.date()
        cutoff = today - ROW_RETENTION
        latest = {day: float(value) for day, value in rows.items() if day >= cutoff}
        if not self.initialized:
            self.rows = latest
            self.baseline_rows = dict(latest)
            self.started_at = now
            return
        for day, value in latest.items():
            delta = value - self.rows.get(day, 0.0)
            self.rows[day] = value
            if day >= today:
                self.total += delta
            elif delta:
                self.late[day] = self.late.get(day, 0.0) + delta
        self.rows = {day: value for day, value in self.rows.items() if day >= cutoff}

    def mark_backdated(self, applied: Mapping[date, float]) -> None:
        """Record that ``applied`` late changes were added to statistics."""
        for day, value in applied.items():
            remaining = self.late.pop(day, 0.0) - value
            if abs(remaining) > 1e-9:
                # More arrived for that day while it was being booked.
                self.late[day] = remaining
            self.backdated += value

    def late_hour(self, day: date, tz: tzinfo) -> datetime:
        """Return the statistics hour a late change to ``day`` belongs in."""
        if self.started_at is not None and day < self.started_at.date():
            return datetime.combine(day, BACKFILL_HOUR, tz)
        return datetime.combine(day, LATE_HOUR, tz)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "total": self.total,
            "rows": _iso_keys(self.rows),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "baseline_rows": _iso_keys(self.baseline_rows),
            "backfilled": self.backfilled,
            "late": _iso_keys(self.late),
            "backdated": self.backdated,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DailyTotal:
        """Build a total from ``as_dict`` output."""
        started_at = data["started_at"]
        return cls(
            total=data["total"],
            rows=_date_keys(data["rows"]),
            started_at=datetime.fromisoformat(started_at) if started_at else None,
            baseline_rows=_date_keys(data["baseline_rows"]),
            backfilled=data["backfilled"],
            late=_date_keys(data["late"]),
            backdated=data["backdated"],
        )


def _iso_keys(rows: Mapping[date, float]) -> dict[str, float]:
    return {day.isoformat(): value for day, value in rows.items()}


def _date_keys(rows: Mapping[str, float]) -> dict[date, float]:
    return {date.fromisoformat(day): value for day, value in rows.items()}


@dataclass
class DrivingMeter:
    """The driving energy (Wh), its parts (Wh) and distance (km) of one vehicle."""

    energy: DailyTotal = field(default_factory=DailyTotal)
    distance: DailyTotal = field(default_factory=DailyTotal)
    energy_motor: DailyTotal = field(default_factory=DailyTotal)
    energy_climate: DailyTotal = field(default_factory=DailyTotal)
    energy_electronics: DailyTotal = field(default_factory=DailyTotal)
    energy_battery_care: DailyTotal = field(default_factory=DailyTotal)
    energy_regenerated: DailyTotal = field(default_factory=DailyTotal)
    odometer_at_start_km: float | None = None
    api_wh_per_km_at_start: float | None = None
    # The API's energy total over a rolling 90-day window. Its per-month
    # breakdown (not exposed by the library) shows the oldest month shrinking
    # day by day, so it is not whole calendar months.
    window_wh: float | None = None
    window_carry_wh: float = 0.0
    # Growth of that total not explained by the daily rows: driving reported
    # for days older than the daily rows, which the meters cannot count.
    missed_wh: float = 0.0

    def daily_total(self, quantity: Quantity) -> DailyTotal:
        """Return the total of ``quantity``."""
        total: DailyTotal = getattr(self, quantity)
        return total

    def update(
        self,
        energy_rows: Mapping[date, float],
        distance_rows: Mapping[date, float],
        now: datetime,
        window_wh: float | None = None,
        odometer_km: float | None = None,
        api_wh_per_km: float | None = None,
    ) -> None:
        """Apply the latest daily rows and 90-day total.

        The odometer and the API's average consumption are kept from the
        energy meter's start, for the lifetime estimate.
        """
        starting = not self.energy.initialized
        measured = self.energy.measured
        self.energy.update(energy_rows, now)
        if starting and self.energy.initialized:
            self.odometer_at_start_km = odometer_km
            self.api_wh_per_km_at_start = api_wh_per_km
        self.distance.update(distance_rows, now)
        if energy_rows and window_wh is not None:
            # Only with daily rows: otherwise their growth is unknown.
            self._update_window(float(window_wh), self.energy.measured - measured)

    def update_parts(
        self, part_rows: Mapping[EnergyPart, Mapping[date, float]], now: datetime
    ) -> None:
        """Apply the latest daily rows of the energy parts."""
        for part, rows in part_rows.items():
            self.daily_total(part).update(rows, now)

    def _update_window(self, window_wh: float, counted_wh: float) -> None:
        """Add growth of the 90-day total beyond ``counted_wh`` to ``missed_wh``.

        The two totals come from separate requests, so a trip reported in
        between shows as a shortfall in one poll and an excess in the next;
        a shortfall is carried to the next poll once. The day dropping out of
        the window each day is a shortfall too, and is carried once as well,
        so a late trip smaller than that day's driving, arriving with the
        drop or in the next poll, goes unnoticed.
        """
        if self.window_wh is not None:
            excess = window_wh - self.window_wh - counted_wh
            self.missed_wh += max(excess + self.window_carry_wh, 0.0)
            self.window_carry_wh = min(excess, 0.0)
        self.window_wh = window_wh

    def lifetime_estimate_wh(self, wh_per_km: float | None) -> float | None:
        """Estimate lifetime driving energy: start odometer x average + measured."""
        if wh_per_km is None:
            wh_per_km = self.api_wh_per_km_at_start
        if (
            not self.energy.initialized
            or self.odometer_at_start_km is None
            or wh_per_km is None
        ):
            return None
        return self.odometer_at_start_km * wh_per_km + self.energy.measured

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "energy": self.energy.as_dict(),
            "distance": self.distance.as_dict(),
            **{part: self.daily_total(part).as_dict() for part in ENERGY_PARTS},
            "odometer_at_start_km": self.odometer_at_start_km,
            "api_wh_per_km_at_start": self.api_wh_per_km_at_start,
            "window_wh": self.window_wh,
            "window_carry_wh": self.window_carry_wh,
            "missed_wh": self.missed_wh,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DrivingMeter:
        """Build a meter from ``as_dict`` output."""
        return cls(
            energy=DailyTotal.from_dict(data["energy"]),
            distance=DailyTotal.from_dict(data["distance"]),
            # A meter saved before the parts existed lacks them: they start now.
            **{
                part: DailyTotal.from_dict(data[part]) if part in data else DailyTotal()
                for part in ENERGY_PARTS
            },
            # Missing from meters saved before the estimate existed.
            odometer_at_start_km=data.get("odometer_at_start_km"),
            api_wh_per_km_at_start=data.get("api_wh_per_km_at_start"),
            window_wh=data["window_wh"],
            window_carry_wh=data["window_carry_wh"],
            missed_wh=data["missed_wh"],
        )


@dataclass
class Backfill:
    """Hourly statistics for the days before a meter started."""

    # (hour, sum) rows; the sums run from the first day's value up to total.
    rows: list[tuple[datetime, float]]
    total: float
    # The first hour the recorder compiles for the meter.
    meter_start: datetime


def backfill_statistics(total: DailyTotal, tz: tzinfo) -> Backfill | None:
    """Return statistics for the days before the meter, or None if unstarted.

    Every row lies strictly before the meter's first hour: importing over
    hours the recorder already compiled would break the sum chain. Home
    Assistant counts the first row's sum as that hour's change, so the sums
    start from the first day's value, as if they had been 0 before.
    """
    if total.started_at is None:
        return None
    meter_start = total.started_at.astimezone(tz).replace(
        minute=0, second=0, microsecond=0
    )
    last_hour = meter_start - timedelta(hours=1)
    rows: list[tuple[datetime, float]] = []
    running = 0.0
    for day, value in sorted(total.baseline_rows.items()):
        # The start day's hour can fall after the meter's first state.
        hour = min(datetime.combine(day, BACKFILL_HOUR, tz), last_hour)
        running += value
        if rows and hour <= rows[-1][0]:
            # Two days in the same hour: one row for both.
            rows[-1] = (rows[-1][0], running)
        else:
            rows.append((hour, running))
    return Backfill(rows, running, meter_start)


def daily_energy_rows(vehicle: Vehicle) -> dict[date, float]:
    """Return the vehicle's driving energy (Wh) per day."""
    return _daily_rows(vehicle, lambda day: day.total_consumed)


def daily_part_rows(vehicle: Vehicle) -> dict[EnergyPart, dict[date, float]]:
    """Return the vehicle's driving energy (Wh) of each part per day."""
    return {
        part: _daily_rows(vehicle, attrgetter(attribute))
        for part, attribute in ENERGY_PARTS.items()
    }


def daily_distance_rows(vehicle: Vehicle) -> dict[date, float]:
    """Return the vehicle's driving distance (km) per day."""

    def km(day: DailyDrivingStats) -> float | None:
        if day.distance is None:
            return None
        if day.distance_unit == LENGTH_MILES:
            return float(day.distance) * KM_PER_MILE
        return float(day.distance)

    return _daily_rows(vehicle, km)


def _daily_rows(
    vehicle: Vehicle, value: Callable[[DailyDrivingStats], float | None]
) -> dict[date, float]:
    rows: dict[date, float] = {}
    for day in vehicle.daily_stats or ():
        if day.date is not None and (amount := value(day)) is not None:
            rows[day.date.date()] = float(amount)
    return rows


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


class DrivingMeterTracker:
    """Keep one vehicle's meter in sync with the vehicle's data.

    Create it with ``async_load``.
    """

    def __init__(
        self,
        store: MeterStore,
        vehicle: Vehicle,
        now: Callable[[], datetime],
        meter: DrivingMeter,
    ) -> None:
        """Initialize the tracker."""
        self._store = store
        self._vehicle = vehicle
        self._now = now
        self._lock = asyncio.Lock()
        self._listeners: list[Callable[[], None]] = []
        self.meter = meter

    @classmethod
    async def async_load(
        cls, store: MeterStore, vehicle: Vehicle, now: Callable[[], datetime]
    ) -> DrivingMeterTracker:
        """Return a tracker for the stored meter, or for a new one."""
        data = await store.async_load()
        if not data:
            _LOGGER.debug("%s: no stored driving meter yet", vehicle.name)
            return cls(store, vehicle, now, DrivingMeter())
        meter = DrivingMeter.from_dict(data)
        for quantity, unit in QUANTITY_UNITS.items():
            total = meter.daily_total(quantity)
            _LOGGER.debug(
                "%s: loaded driving %s meter: total %s %s since %s, %d days of "
                "rows, backfilled %s, late %s, booked late so far %s",
                vehicle.name,
                quantity,
                total.total,
                unit,
                total.started_at,
                len(total.rows),
                total.backfilled,
                _iso_keys(total.late),
                total.backdated,
            )
        return cls(store, vehicle, now, meter)

    async def async_process(self) -> None:
        """Apply the vehicle's latest daily rows, then notify the listeners."""
        energy_rows = daily_energy_rows(self._vehicle)
        distance_rows = daily_distance_rows(self._vehicle)
        window_wh = self._vehicle.total_power_consumed
        odometer = odometer_km(self._vehicle)
        api_wh_per_km = self._vehicle.power_consumption_30d
        part_rows = daily_part_rows(self._vehicle)
        now = self._now()

        def update(meter: DrivingMeter) -> None:
            meter.update(
                energy_rows, distance_rows, now, window_wh, odometer, api_wh_per_km
            )
            meter.update_parts(part_rows, now)

        before = await self._async_change(update)
        if before is None:
            _LOGGER.debug(
                "%s: driving meters unchanged (%d daily rows)",
                self._vehicle.name,
                len(energy_rows),
            )
            return
        self._log_changes(before)
        if (missed := self.meter.missed_wh - before.missed_wh) > 0:
            _LOGGER.warning(
                "%s: %.0f Wh of driving was reported for days older than the "
                "~30 days of daily rows. It is not in Driving Energy or Driving "
                "Distance (%.0f Wh missed so far)",
                self._vehicle.name,
                missed,
                self.meter.missed_wh,
            )
        for listener in list(self._listeners):
            listener()

    def _log_changes(self, before: DrivingMeter) -> None:
        """Log what one poll changed."""
        for quantity, unit in QUANTITY_UNITS.items():
            old = before.daily_total(quantity)
            new = self.meter.daily_total(quantity)
            if not old.initialized:
                if new.initialized:
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
                day.isoformat(): value - old.rows.get(day, 0.0)
                for day, value in sorted(new.rows.items())
                if old.rows.get(day) != value
            }
            dropped = sorted(day.isoformat() for day in old.rows.keys() - new.rows)
            if changes or dropped:
                _LOGGER.debug(
                    "%s: driving %s rows changed by %s %s, dropped %s; "
                    "total %s -> %s, late %s",
                    self._vehicle.name,
                    quantity,
                    changes,
                    unit,
                    dropped,
                    old.total,
                    new.total,
                    _iso_keys(new.late),
                )
        if self.meter.window_wh != before.window_wh:
            _LOGGER.debug(
                "%s: 90-day driving energy total %s -> %s Wh, carried %s Wh",
                self._vehicle.name,
                before.window_wh,
                self.meter.window_wh,
                self.meter.window_carry_wh,
            )

    async def async_mark_backfilled(self, quantity: Quantity) -> None:
        """Record that the days before ``quantity``'s start were imported."""

        def mark(meter: DrivingMeter) -> None:
            meter.daily_total(quantity).backfilled = True

        await self._async_change(mark)

    async def async_mark_backdated(
        self, quantity: Quantity, applied: Mapping[date, float]
    ) -> None:
        """Record that ``applied`` late changes were added to statistics."""
        await self._async_change(
            lambda meter: meter.daily_total(quantity).mark_backdated(applied)
        )

    async def _async_change(
        self, change: Callable[[DrivingMeter], None]
    ) -> DrivingMeter | None:
        """Apply ``change`` to a copy of the meter; save it if it differs.

        Return the meter as it was before, or None if nothing changed. The
        meter is replaced, never changed in place, so earlier references to
        it keep their state.
        """
        async with self._lock:
            before = self.meter
            updated = copy.deepcopy(before)
            change(updated)
            if updated == before:
                return None
            # Persist first: a crash after publishing but before saving would
            # restore a lower total, which statistics book as negative usage.
            await self._store.async_save(updated.as_dict())
            self.meter = updated
            return before

    def add_listener(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call ``listener`` after each published change; return a remover."""
        self._listeners.append(listener)

        def remove() -> None:
            self._listeners.remove(listener)

        return remove
