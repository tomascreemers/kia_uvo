"""Sensor for Hyundai / Kia Connect integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import partial
import logging
from typing import Any, Final, Literal, cast

from homeassistant.components.recorder.models import (
    StatisticData,
    StatisticMeanType,
    StatisticMetaData,
)
from homeassistant.components.recorder.statistics import (
    StatisticsRow,
    async_import_statistics,
    get_last_short_term_statistics,
    get_last_statistics,
    get_metadata,
    statistics_during_period,
)
from homeassistant.components.sensor import (
    RestoreSensor,
    SensorDeviceClass,
    SensorEntity,
    SensorEntityDescription,
    SensorStateClass,
)
from homeassistant.components.sensor.const import UNIT_CONVERTERS
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import (
    PERCENTAGE,
    EntityCategory,
    UnitOfElectricPotential,
    UnitOfEnergy,
    UnitOfLength,
    UnitOfPower,
    UnitOfTime,
)
from homeassistant.core import CoreState, HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.recorder import get_instance
from homeassistant.helpers.start import async_at_started
from homeassistant.helpers.typing import StateType
from homeassistant.util import dt as dt_util
from hyundai_kia_connect_api import Vehicle
from hyundai_kia_connect_api.const import ENGINE_TYPES

from .const import CHARGING_CURRENTS, DOMAIN, DYNAMIC_UNIT
from .coordinator import HyundaiKiaConnectDataUpdateCoordinator
from .driving_energy import (
    ROW_RETENTION,
    DailyTotal,
    DrivingMeterTracker,
    LateChange,
    Quantity,
    backfill_statistics,
)
from .entity import HyundaiKiaConnectEntity

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HyundaiKiaSensorEntityDescription(SensorEntityDescription):
    """A class that describes custom sensor entities."""

    exists: Callable[[Vehicle], bool] | None = None


def _is_electrified(vehicle: Vehicle) -> bool:
    """Return True for BEV and PHEV vehicles."""
    return vehicle.engine_type in (ENGINE_TYPES.EV, ENGINE_TYPES.PHEV)


SENSOR_DESCRIPTIONS: Final[tuple[HyundaiKiaSensorEntityDescription, ...]] = (
    HyundaiKiaSensorEntityDescription(
        key="_total_driving_range",
        translation_key="total_driving_range",
        icon="mdi:road-variant",
        device_class=SensorDeviceClass.DISTANCE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="_odometer",
        translation_key="odometer",
        icon="mdi:speedometer",
        native_unit_of_measurement=DYNAMIC_UNIT,
        device_class=SensorDeviceClass.DISTANCE,
        state_class=SensorStateClass.TOTAL_INCREASING,
        # Odometer is a stable vehicle capability, but its value can be absent
        # during setup. Keep the entity so a later refresh can populate it.
        exists=lambda _: True,
    ),
    HyundaiKiaSensorEntityDescription(
        key="_last_service_distance",
        translation_key="last_service_distance",
        icon="mdi:car-wrench",
        device_class=SensorDeviceClass.DISTANCE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="_next_service_distance",
        translation_key="next_service_distance",
        icon="mdi:car-wrench",
        device_class=SensorDeviceClass.DISTANCE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="car_battery_percentage",
        translation_key="car_battery_percentage",
        icon="mdi:car-battery",
        native_unit_of_measurement=PERCENTAGE,
        device_class=SensorDeviceClass.BATTERY,
        state_class=SensorStateClass.MEASUREMENT,
        # The 12V SoC is transient — None while the telematics unit
        # is asleep, after a 12V reset, or when the status payload
        # omits it. Don't gate creation on it: a None at setup (e.g.
        # a version-update reload) means the entity isn't yielded and
        # HA marks it "no longer provided", with no return until the
        # next reload. Always create; None -> HA `unknown`, the real
        # SoC arrives on the next poll. See #1803.
        exists=lambda _: True,
    ),
    HyundaiKiaSensorEntityDescription(
        key="last_updated_at",
        translation_key="last_updated_at",
        icon="mdi:update",
        device_class=SensorDeviceClass.TIMESTAMP,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="last_scanned_at",
        translation_key="last_scanned_at",
        icon="mdi:cloud-search",
        device_class=SensorDeviceClass.TIMESTAMP,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_battery_percentage",
        translation_key="ev_battery_percentage",
        native_unit_of_measurement=PERCENTAGE,
        device_class=SensorDeviceClass.BATTERY,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_battery_soh_percentage",
        translation_key="ev_battery_soh_percentage",
        native_unit_of_measurement=PERCENTAGE,
        device_class=SensorDeviceClass.BATTERY,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_battery_remain",
        translation_key="ev_battery_remain",
        native_unit_of_measurement=UnitOfEnergy.KILO_JOULE,
        device_class=SensorDeviceClass.ENERGY_STORAGE,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_battery_capacity",
        translation_key="ev_battery_capacity",
        native_unit_of_measurement=UnitOfEnergy.KILO_JOULE,
        device_class=SensorDeviceClass.ENERGY_STORAGE,
        state_class=SensorStateClass.MEASUREMENT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="_ev_driving_range",
        translation_key="ev_driving_range",
        icon="mdi:road-variant",
        device_class=SensorDeviceClass.DISTANCE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="_fuel_driving_range",
        translation_key="fuel_driving_range",
        icon="mdi:road-variant",
        device_class=SensorDeviceClass.DISTANCE,
        native_unit_of_measurement=DYNAMIC_UNIT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="fuel_level",
        translation_key="fuel_level",
        native_unit_of_measurement=PERCENTAGE,
        icon="mdi:fuel",
        state_class=SensorStateClass.MEASUREMENT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="_air_temperature",
        translation_key="air_temperature",
        native_unit_of_measurement=DYNAMIC_UNIT,
        device_class=SensorDeviceClass.TEMPERATURE,
        # The setpoint is transient — it is None while climate is off
        # (USA returns airTemp.value "OFF"), so don't gate on it. Gate
        # on climate presence (air_control_is_on, the same signal the
        # climate entity uses) to avoid creating an unusable sensor on
        # vehicles that report no climate. A None setpoint -> HA
        # `unknown`; the real setpoint arrives on the next poll.
        exists=lambda vehicle: (
            vehicle.air_control_is_on is not None
            or vehicle._air_temperature is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_estimated_current_charge_duration",
        translation_key="ev_estimated_current_charge_duration",
        icon="mdi:ev-station",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.MINUTES,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_estimated_fast_charge_duration",
        translation_key="ev_estimated_fast_charge_duration",
        icon="mdi:ev-station",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.MINUTES,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_estimated_portable_charge_duration",
        translation_key="ev_estimated_portable_charge_duration",
        icon="mdi:ev-station",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.MINUTES,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_estimated_station_charge_duration",
        translation_key="ev_estimated_station_charge_duration",
        icon="mdi:ev-station",
        device_class=SensorDeviceClass.DURATION,
        native_unit_of_measurement=UnitOfTime.MINUTES,
        state_class=SensorStateClass.MEASUREMENT,
    ),
    # Target charge range is transient — None at setup when the car is
    # asleep, telematics omits dte.rangeByFuel.totalAvailableRange, or
    # targetSOClist has a single entry (one of the two indices is then
    # absent). Value-gating here drops the entity permanently ("no longer
    # provided") on a version-update reload, and it only returns on a
    # manual reload while the value happens to be present. Gate on the
    # stable EV/PHEV capability instead so a later coordinator poll can
    # publish the range. None -> HA `unknown` until the next poll. See
    # #1842. The same gate applies to both sensors below.
    HyundaiKiaSensorEntityDescription(
        key="_ev_target_range_charge_AC",
        translation_key="ev_target_range_charge_ac",
        icon="mdi:ev-station",
        device_class=SensorDeviceClass.DISTANCE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        exists=lambda vehicle: (
            _is_electrified(vehicle) or vehicle._ev_target_range_charge_AC is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="_ev_target_range_charge_DC",
        translation_key="ev_target_range_charge_dc",
        icon="mdi:ev-station",
        device_class=SensorDeviceClass.DISTANCE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        exists=lambda vehicle: (
            _is_electrified(vehicle) or vehicle._ev_target_range_charge_DC is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="total_power_consumed",
        translation_key="total_power_consumed",
        icon="mdi:car-electric",
        native_unit_of_measurement=UnitOfEnergy.WATT_HOUR,
        device_class=SensorDeviceClass.ENERGY,
        # A rolling 90-day window shrinks as days drop out; with a state
        # class, statistics would book every shrink as negative usage.
        # driving_energy is the statistics-safe counterpart.
    ),
    HyundaiKiaSensorEntityDescription(
        key="total_power_regenerated",
        translation_key="total_power_regenerated",
        icon="mdi:car-electric",
        native_unit_of_measurement=UnitOfEnergy.WATT_HOUR,
        device_class=SensorDeviceClass.ENERGY,
        # Rolling 90-day window, see total_power_consumed.
    ),
    # Need to remove km hard coding.  Underlying API needs this fixed first.  EU always does KM.
    HyundaiKiaSensorEntityDescription(
        key="power_consumption_30d",
        translation_key="power_consumption_30d",
        icon="mdi:car-electric",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=f"{UnitOfEnergy.WATT_HOUR}/km",
    ),
    HyundaiKiaSensorEntityDescription(
        key="front_left_seat_status",
        translation_key="front_left_seat_status",
        icon="mdi:car-seat-heater",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="front_right_seat_status",
        translation_key="front_right_seat_status",
        icon="mdi:car-seat-heater",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="rear_left_seat_status",
        translation_key="rear_left_seat_status",
        icon="mdi:car-seat-heater",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="rear_right_seat_status",
        translation_key="rear_right_seat_status",
        icon="mdi:car-seat-heater",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="_geocode_name",
        translation_key="geocode_name",
        icon="mdi:map",
        # The geocoded address is transient — it lives in memory and is None
        # right after a restart/reload until a poll with coordinates
        # succeeds. Don't gate creation on it: a None at setup means the
        # entity isn't yielded and HA marks it "no longer provided", with no
        # return until the next reload, even after location recovers
        # (kia_uvo #1844). Always create; None -> HA `unknown`.
        exists=lambda _: True,
    ),
    HyundaiKiaSensorEntityDescription(
        key="dtc_count",
        translation_key="dtc_count",
        icon="mdi:alert-circle",
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_first_departure_time",
        translation_key="ev_first_departure_time",
        icon="mdi:clock-outline",
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_second_departure_time",
        translation_key="ev_second_departure_time",
        icon="mdi:clock-outline",
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_off_peak_start_time",
        translation_key="ev_off_peak_start_time",
        icon="mdi:clock-outline",
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_off_peak_end_time",
        translation_key="ev_off_peak_end_time",
        icon="mdi:clock-outline",
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_charging_current",
        translation_key="ev_charging_current",
        icon="mdi:lightning-bolt-circle",
        native_unit_of_measurement=PERCENTAGE,
        device_class=SensorDeviceClass.POWER_FACTOR,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_charging_power",
        translation_key="ev_charging_power",
        icon="mdi:flash",
        native_unit_of_measurement=UnitOfPower.KILO_WATT,
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        # Charging power is transient and usually None at setup while
        # the vehicle is unplugged. Create the sensor for electrified
        # vehicles so a later coordinator poll can publish the value
        # without requiring an integration reload.
        exists=lambda vehicle: (
            _is_electrified(vehicle) or vehicle.ev_charging_power is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="VIN",
        translation_key="vehicle_identification_number",
        icon="mdi:identifier",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="_outside_temperature",
        translation_key="outside_temperature",
        icon="mdi:thermometer",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=DYNAMIC_UNIT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="engine_type",
        translation_key="engine_type",
        icon="mdi:engine",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_battery_chiller_rpm",
        translation_key="ev_battery_chiller_rpm",
        icon="mdi:fan",
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement="rpm",
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_first_departure_days",
        translation_key="ev_first_departure_days",
        icon="mdi:calendar-clock",
        entity_category=EntityCategory.DIAGNOSTIC,
        exists=lambda vehicle: (
            getattr(vehicle, "ev_first_departure_enabled", None) is not None
            or getattr(vehicle, "ev_first_departure_time", None) is not None
            or getattr(vehicle, "ev_first_departure_days", None) is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_second_departure_days",
        translation_key="ev_second_departure_days",
        icon="mdi:calendar-clock",
        entity_category=EntityCategory.DIAGNOSTIC,
        exists=lambda vehicle: (
            getattr(vehicle, "ev_second_departure_enabled", None) is not None
            or getattr(vehicle, "ev_second_departure_time", None) is not None
            or getattr(vehicle, "ev_second_departure_days", None) is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="_ev_first_departure_climate_temperature",
        translation_key="ev_first_departure_climate_temperature",
        icon="mdi:thermometer",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=DYNAMIC_UNIT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="_ev_second_departure_climate_temperature",
        translation_key="ev_second_departure_climate_temperature",
        icon="mdi:thermometer",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=DYNAMIC_UNIT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_battery_pack_voltage",
        translation_key="ev_battery_pack_voltage",
        icon="mdi:car-battery",
        device_class=SensorDeviceClass.VOLTAGE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfElectricPotential.VOLT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_battery_temperature_min",
        translation_key="ev_battery_temperature_min",
        icon="mdi:thermometer-low",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=DYNAMIC_UNIT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_battery_temperature_max",
        translation_key="ev_battery_temperature_max",
        icon="mdi:thermometer-high",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=DYNAMIC_UNIT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_battery_water_temperature",
        translation_key="ev_battery_water_temperature",
        icon="mdi:thermometer-water",
        device_class=SensorDeviceClass.TEMPERATURE,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=DYNAMIC_UNIT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_power_consumption_air_conditioning",
        translation_key="ev_power_consumption_air_conditioning",
        icon="mdi:air-conditioner",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_power_consumption_battery_cooling",
        translation_key="ev_power_consumption_battery_cooling",
        icon="mdi:snowflake",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="ev_power_consumption_battery_heater",
        translation_key="ev_power_consumption_battery_heater",
        icon="mdi:radiator",
        device_class=SensorDeviceClass.POWER,
        state_class=SensorStateClass.MEASUREMENT,
        native_unit_of_measurement=UnitOfPower.WATT,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    HyundaiKiaSensorEntityDescription(
        key="location_last_updated_at",
        translation_key="location_last_updated_at",
        icon="mdi:map-clock",
        device_class=SensorDeviceClass.TIMESTAMP,
        entity_category=EntityCategory.DIAGNOSTIC,
    ),
    # Transient like _air_temperature above: some backends (AU/NZ)
    # report the TPMS no-data sentinel whenever the car is parked
    # — nearly always the case at setup — so don't gate on the
    # value. The parsed unit is the capability signal: non-None
    # exactly for direct-TPMS vehicles (known PressureUnit), None
    # for indirect TPMS (PressureUnit 3, e.g. KONA — #1786) and
    # old-protocol vehicles, which never report a numeric
    # pressure. A None pressure -> HA `unknown` until a poll
    # catches the car driving. The same gate applies to all four
    # tire_pressure_* sensors below.
    HyundaiKiaSensorEntityDescription(
        key="tire_pressure_front_left",
        translation_key="tire_pressure_front_left",
        device_class=SensorDeviceClass.PRESSURE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        state_class=SensorStateClass.MEASUREMENT,
        exists=lambda vehicle: (
            vehicle.tire_pressure_front_left is not None
            or vehicle.tire_pressure_unit is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="tire_pressure_front_right",
        translation_key="tire_pressure_front_right",
        device_class=SensorDeviceClass.PRESSURE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        state_class=SensorStateClass.MEASUREMENT,
        exists=lambda vehicle: (
            vehicle.tire_pressure_front_right is not None
            or vehicle.tire_pressure_unit is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="tire_pressure_rear_left",
        translation_key="tire_pressure_rear_left",
        device_class=SensorDeviceClass.PRESSURE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        state_class=SensorStateClass.MEASUREMENT,
        exists=lambda vehicle: (
            vehicle.tire_pressure_rear_left is not None
            or vehicle.tire_pressure_unit is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="tire_pressure_rear_right",
        translation_key="tire_pressure_rear_right",
        device_class=SensorDeviceClass.PRESSURE,
        native_unit_of_measurement=DYNAMIC_UNIT,
        state_class=SensorStateClass.MEASUREMENT,
        exists=lambda vehicle: (
            vehicle.tire_pressure_rear_right is not None
            or vehicle.tire_pressure_unit is not None
        ),
    ),
    HyundaiKiaSensorEntityDescription(
        key="drive_mode",
        translation_key="drive_mode",
        icon="mdi:car-cog",
    ),
)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensor platform."""
    coordinator = hass.data[DOMAIN][config_entry.unique_id]
    entities: list[SensorEntity] = []
    for vehicle_id in coordinator.vehicle_manager.vehicles:
        vehicle: Vehicle = coordinator.vehicle_manager.vehicles[vehicle_id]
        for description in SENSOR_DESCRIPTIONS:
            if (
                description.key == "_geocode_name"
                and not coordinator.vehicle_manager.geocode_api_enable
            ):
                # Created only when the geolocation option is on: without it
                # the library never runs the geocode lookup, so the sensor
                # would sit `unknown` forever. With the option on, a None at
                # setup must not gate creation — see the `_geocode_name`
                # description above. Gated here, not in `exists`, because the
                # option is coordinator state.
                continue
            create = (
                description.exists(vehicle)
                if description.exists is not None
                else getattr(vehicle, description.key, None) is not None
            )
            if create:
                entities.append(
                    HyundaiKiaConnectSensor(coordinator, description, vehicle)
                )
        if vehicle.daily_stats:
            entities.append(
                DailyDrivingStatsEntity(
                    coordinator, coordinator.vehicle_manager.vehicles[vehicle_id]
                )
            )
            entities.append(
                TodaysDailyDrivingStatsEntity(
                    coordinator, coordinator.vehicle_manager.vehicles[vehicle_id]
                )
            )
        if (tracker := coordinator.driving_meters.get(vehicle_id)) is not None:
            entities.extend(
                _driving_meter_sensors(coordinator, config_entry, vehicle, tracker)
            )
        entities.append(
            VehicleEntity(coordinator, coordinator.vehicle_manager.vehicles[vehicle_id])
        )
    for vehicle_id in coordinator.vehicle_manager.vehicles:
        if await coordinator.async_supports_svm(vehicle_id):
            entities.append(
                SVMStatusSensor(
                    coordinator, coordinator.vehicle_manager.vehicles[vehicle_id]
                )
            )
    async_add_entities(entities)


PARALLEL_UPDATES = 0


@dataclass(frozen=True, kw_only=True)
class DrivingMeterSensorEntityDescription(SensorEntityDescription):
    """Describes a driving meter sensor."""

    quantity: Quantity
    native_unit_of_measurement: str


DRIVING_METER_DESCRIPTIONS: Final[tuple[DrivingMeterSensorEntityDescription, ...]] = (
    DrivingMeterSensorEntityDescription(
        key="driving_energy",
        translation_key="driving_energy",
        quantity="energy",
        icon="mdi:car-electric",
        device_class=SensorDeviceClass.ENERGY,
        state_class=SensorStateClass.TOTAL,
        native_unit_of_measurement=UnitOfEnergy.WATT_HOUR,
        suggested_unit_of_measurement=UnitOfEnergy.KILO_WATT_HOUR,
        suggested_display_precision=2,
    ),
    # Built from the same daily rows as the energy, so both cover the same
    # trips and are booked on the same days. The API reports whole kilometres
    # per day.
    DrivingMeterSensorEntityDescription(
        key="driving_distance",
        translation_key="driving_distance",
        quantity="distance",
        icon="mdi:map-marker-distance",
        device_class=SensorDeviceClass.DISTANCE,
        state_class=SensorStateClass.TOTAL,
        native_unit_of_measurement=UnitOfLength.KILOMETERS,
        suggested_display_precision=0,
    ),
)


def _driving_meter_sensors(
    coordinator: HyundaiKiaConnectDataUpdateCoordinator,
    config_entry: ConfigEntry,
    vehicle: Vehicle,
    tracker: DrivingMeterTracker,
) -> list[SensorEntity]:
    """Return the driving meters."""
    return [
        DrivingMeterSensor(coordinator, config_entry, vehicle, tracker, description)
        for description in DRIVING_METER_DESCRIPTIONS
    ]


class DrivingMeterSensor(SensorEntity, HyundaiKiaConnectEntity):
    """A total built from the daily driving rows, for statistics.

    Every change moves the state when it is reported, and the recorder books
    it in the statistics of that moment. A change to an earlier day is then
    moved to that day's statistics, so the sums show it on the day it was
    driven. The states cannot be moved: the history graph shows it when it
    was reported.

    The state starts at the total of the days before the meter, which are
    imported as statistics leading up to it.
    """

    entity_description: DrivingMeterSensorEntityDescription

    def __init__(
        self,
        coordinator: HyundaiKiaConnectDataUpdateCoordinator,
        config_entry: ConfigEntry,
        vehicle: Vehicle,
        tracker: DrivingMeterTracker,
        description: DrivingMeterSensorEntityDescription,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, vehicle)
        self.entity_description = description
        self._attr_unique_id = f"{DOMAIN}_{vehicle.id}_{description.key}"
        self._config_entry = config_entry
        self._tracker = tracker

    @property
    def _daily_total(self) -> DailyTotal:
        return self._tracker.meter.daily_total(self.entity_description.quantity)

    @property
    def native_value(self) -> float | None:
        """Return the meter reading, or None until the meter has started."""
        total = self._daily_total
        return total.reading if total.initialized else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return what was booked outside the state."""
        total = self._daily_total
        if total.started_at is None:
            return {}
        unit = self.entity_description.native_unit_of_measurement.lower()
        return {
            "measured_since": total.started_at.date().isoformat(),
            f"booked_on_earlier_days_{unit}": round(total.measured - total.total, 3),
        }

    async def async_added_to_hass(self) -> None:
        """Publish whenever the tracker has saved a change."""
        await super().async_added_to_hass()
        self.async_on_remove(self._tracker.add_listener(self._handle_meter_change))
        self.async_on_remove(async_at_started(self.hass, self._async_at_started))

    @callback
    def _async_at_started(self, hass: HomeAssistant) -> None:
        self._schedule_statistics_update()

    @callback
    def _handle_meter_change(self) -> None:
        self.async_write_ha_state()
        self._schedule_statistics_update()

    @callback
    def _handle_coordinator_update(self) -> None:
        super()._handle_coordinator_update()
        # Retries statistics work that waited for the recorder.
        self._schedule_statistics_update()

    @callback
    def _schedule_statistics_update(self) -> None:
        # The recorder commits nothing before Home Assistant has started, and
        # the start waits for tasks like this one.
        if self.hass.state is not CoreState.running:
            return
        if self._daily_total.statistics_pending:
            # Not a background task: cancelled after the recorder committed
            # but before the mark, the change would be booked again.
            self._config_entry.async_create_task(
                self.hass,
                self._async_update_statistics(),
                f"{DOMAIN} {self.entity_id} statistics",
            )

    async def _async_update_statistics(self) -> None:
        """Import the days before the meter once, then book the late changes.

        Both go into this sensor's own statistics, as changes to the sums
        from a given hour onwards. The recorder continues the sum from the
        sensor's last short-term statistics, so this waits until it has
        compiled some: a change made before then would be lost at the next
        compile. The recorder runs its jobs in order, so each change lands on
        top of the previous ones.

        A change is only marked as done once the recorder has committed it:
        marked earlier, a crash in between would lose it for good. A crash
        after the commit but before the mark is saved books it twice; that
        window is a single file write.
        """
        if "recorder" not in self.hass.config.components:
            return
        quantity = self.entity_description.quantity
        async with self._tracker.statistics_lock(quantity):
            # A snapshot: the tracker replaces its meter on every change.
            total = self._daily_total
            if not total.statistics_pending:
                return
            compiled = await self._async_last_short_term_start()
            if compiled is None:
                _LOGGER.debug(
                    "%s has no short-term statistics yet; its backfill and late "
                    "changes wait for them",
                    self.entity_id,
                )
                return
            recorder = get_instance(self.hass)
            if not total.backfilled:
                await self._async_import_backfill(total)
                await recorder.async_block_till_done()
                await self._tracker.async_mark_backfilled(quantity)
            for change in total.late:
                # A late change can only be taken out of the statistics of the
                # period it was reported in once the recorder has compiled it.
                if _short_term_period(change.reported_at).timestamp() > compiled:
                    continue
                if (hour := await self._async_booking_hour(total, change)) is None:
                    continue
                self._move_late(change, hour)
                # One at a time: the next one's hour is looked up in the
                # statistics this one changes.
                await recorder.async_block_till_done()
                await self._tracker.async_mark_backdated(quantity, [change])

    async def _async_last_short_term_start(self) -> float | None:
        """Return when the last compiled short-term statistics start, if any."""
        last = await get_instance(self.hass).async_add_executor_job(
            get_last_short_term_statistics,
            self.hass,
            1,
            self.entity_id,
            False,
            {"sum"},
        )
        if rows := last.get(self.entity_id):
            return rows[0]["start"]
        return None

    async def _async_last_hour_start(self) -> float | None:
        """Return when the last compiled hourly statistics start, if any."""
        last = await get_instance(self.hass).async_add_executor_job(
            get_last_statistics, self.hass, 1, self.entity_id, False, {"sum"}
        )
        if rows := last.get(self.entity_id):
            return rows[0]["start"]
        return None

    async def _async_statistics_rows(
        self, start: datetime, end: datetime, period: Literal["hour", "day"]
    ) -> list[StatisticsRow]:
        """Return the statistics from ``start`` up to ``end``, oldest first."""
        rows = await get_instance(self.hass).async_add_executor_job(
            statistics_during_period,
            self.hass,
            start,
            end,
            {self.entity_id},
            period,
            None,
            {"state", "sum"},
        )
        return rows.get(self.entity_id, [])

    async def _async_statistics_metadata(self) -> StatisticMetaData:
        """Return metadata to import with, in the unit the statistics are kept in.

        The recorder keeps the unit of the first compiled state; importing
        with another one would relabel the rows already there.
        """
        converter = UNIT_CONVERTERS[self.device_class]
        stored = await get_instance(self.hass).async_add_executor_job(
            partial(get_metadata, self.hass, statistic_ids={self.entity_id})
        )
        unit = (
            stored[self.entity_id][1]["unit_of_measurement"]
            if self.entity_id in stored
            else self.unit_of_measurement
        )
        return {
            "mean_type": StatisticMeanType.NONE,
            "has_sum": True,
            "name": None,
            "source": "recorder",
            "statistic_id": self.entity_id,
            "unit_class": converter.UNIT_CLASS,
            "unit_of_measurement": unit,
        }

    async def _async_import_backfill(self, total: DailyTotal) -> None:
        """Import the days before the meter, then shift the compiled sums.

        The recorder started this sensor's sums at 0, so the sums compiled
        since the meter started are shifted up by the imported total. Each
        imported state equals its sum: the state the meter started from is
        that total too.
        Re-importing replaces rows of an earlier import at the same hours.
        """
        backfill = backfill_statistics(total, dt_util.get_default_time_zone())
        if backfill is None or not backfill.rows:
            return
        native_unit = self.entity_description.native_unit_of_measurement
        metadata = await self._async_statistics_metadata()
        unit = metadata["unit_of_measurement"]
        converter = UNIT_CONVERTERS[self.device_class]
        statistics: list[StatisticData] = []
        for hour, value in backfill.rows:
            converted = converter.convert(value, native_unit, unit)
            statistics.append({"start": hour, "state": converted, "sum": converted})
        async_import_statistics(self.hass, metadata, statistics)
        get_instance(self.hass).async_adjust_statistics(
            self.entity_id, backfill.meter_start, backfill.total, native_unit
        )
        _LOGGER.info(
            "Imported %d days of driving %s before %s into %s",
            len(total.baseline_rows),
            self.entity_description.quantity,
            total.started_at,
            self.entity_id,
        )

    async def _async_booking_hour(
        self, total: DailyTotal, change: LateChange
    ) -> datetime | None:
        """Return the hour to move ``change`` to, or None to wait.

        That is its day's hour for late changes if the recorder compiled it,
        else the latest hour of that day it did compile (the sensor was
        unavailable after it), so it never lands on another day. A day with
        no hour at all gets a row at that hour, carrying the sum before it.
        Until the recorder has compiled past that hour, it waits.
        """
        tz = dt_util.get_default_time_zone()
        hour = total.late_hour(change.day, tz)
        last = await self._async_last_hour_start()
        if last is None or last < hour.timestamp():
            return None
        day_start = dt_util.as_utc(datetime.combine(change.day, time(), tz))
        end = hour + timedelta(hours=1)
        if rows := await self._async_statistics_rows(day_start, end, "hour"):
            return dt_util.utc_from_timestamp(rows[-1]["start"])
        # By day: each day's row holds its last hour's sum and state.
        before = await self._async_statistics_rows(hour - ROW_RETENTION, hour, "day")
        if before and (previous := before[-1])["sum"] is not None:
            row: StatisticData = {"start": hour, "sum": previous["sum"]}
            if previous["state"] is not None:
                row["state"] = previous["state"]
            async_import_statistics(
                self.hass, await self._async_statistics_metadata(), [row]
            )
        return hour

    def _move_late(self, change: LateChange, hour: datetime) -> None:
        """Move a late change from when it was reported to ``hour``."""
        recorder = get_instance(self.hass)
        native_unit = self.entity_description.native_unit_of_measurement
        reported = _short_term_period(change.reported_at)
        recorder.async_adjust_statistics(
            self.entity_id, hour, change.value, native_unit
        )
        recorder.async_adjust_statistics(
            self.entity_id, reported, -change.value, native_unit
        )
        _LOGGER.info(
            "Moved %s %s of driving %s for %s, reported late at %s, to %s in %s",
            change.value,
            native_unit,
            self.entity_description.quantity,
            change.day,
            reported.isoformat(),
            hour.isoformat(),
            self.entity_id,
        )


def _short_term_period(moment: datetime) -> datetime:
    """Return the start of the recorder's 5-minute period holding ``moment``.

    The meter takes the time just before it saves and publishes a change, so
    the state moves in that period unless the save straddles its end.
    """
    moment = dt_util.as_utc(moment)
    return moment.replace(
        minute=moment.minute - moment.minute % 5, second=0, microsecond=0
    )


class HyundaiKiaConnectSensor(RestoreSensor, SensorEntity, HyundaiKiaConnectEntity):
    """Hyundai / Kia Connect sensor class."""

    def __init__(
        self,
        coordinator: HyundaiKiaConnectDataUpdateCoordinator,
        description: HyundaiKiaSensorEntityDescription,
        vehicle: Vehicle,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, vehicle)
        self.entity_description = description
        self._key = description.key
        self._last_dynamic_unit: str | None = None
        self._attr_unique_id = f"{DOMAIN}_{vehicle.id}_{self._key}"
        self._attr_icon = description.icon
        self._attr_state_class = description.state_class
        self._attr_device_class = description.device_class
        if description.entity_category:
            self._attr_entity_category = description.entity_category
        # For electrified vehicles (BEV/PHEV) the traction battery is the
        # device's primary battery. Drop the battery device_class from the
        # 12 V auxiliary sensor so Home Assistant's device-page battery picker
        # (which selects the first sensor with device_class=battery, ignoring
        # entity_category) shows the EV battery instead of the 12 V level.
        # HEV/ICE keep the 12 V as their battery. See issue #1749.
        if description.key == "car_battery_percentage":
            engine_type = getattr(vehicle, "engine_type", None)
            if engine_type in (ENGINE_TYPES.EV, ENGINE_TYPES.PHEV):
                self._attr_device_class = None

    @property
    def native_value(self) -> StateType | datetime:
        """Return the value reported by the sensor."""
        value = getattr(self.vehicle, self._key)
        if self._key == "ev_charging_current":
            return CHARGING_CURRENTS.get(value, None)
        if self._key in ("ev_first_departure_days", "ev_second_departure_days"):
            if isinstance(value, list) and value:
                return ", ".join(str(d) for d in value)
            if value in (None, [], ""):
                return "None"
            return cast(StateType | datetime, value)
        return cast(StateType | datetime, value)

    async def async_added_to_hass(self) -> None:
        """Restore the last known dynamic unit after a restart."""
        await super().async_added_to_hass()
        last_sensor_data = await self.async_get_last_sensor_data()
        if last_sensor_data is not None and last_sensor_data.native_unit_of_measurement:
            self._last_dynamic_unit = last_sensor_data.native_unit_of_measurement

    @property
    def native_unit_of_measurement(self) -> str | None:
        """Return the reported unit, falling back to the last known unit.

        A poll whose payload omits the unit block would otherwise publish a
        unitless state for DYNAMIC_UNIT sensors while the device_class stays,
        which trips the recorder units_changed repair on every dropout
        (issue #1894).
        """
        if self.entity_description.native_unit_of_measurement == DYNAMIC_UNIT:
            unit = cast(str | None, getattr(self.vehicle, self._key + "_unit", None))
            if unit is not None:
                self._last_dynamic_unit = unit
                return unit
            return self._last_dynamic_unit
        return self.entity_description.native_unit_of_measurement

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the geocoded address or DTC descriptions, if applicable."""
        if self.entity_description.key == "_geocode_name":
            return {"address": self.vehicle._geocode_address}
        elif self.entity_description.key == "dtc_count":
            return {"DTC Text": self.vehicle.dtc_descriptions}
        return {}


class VehicleEntity(SensorEntity, HyundaiKiaConnectEntity):
    """Sensor exposing the raw vehicle data as attributes."""

    _attr_translation_key = "data"

    def __init__(
        self,
        coordinator: HyundaiKiaConnectDataUpdateCoordinator,
        vehicle: Vehicle,
    ) -> None:
        """Initialize the vehicle data sensor."""
        super().__init__(coordinator, vehicle)

    @property
    def native_value(self) -> StateType:
        """Return a constant state; the data lives in the attributes."""
        return "on"

    @property
    def is_on(self) -> bool:
        """Return True; this sensor is always on."""
        return True

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the raw vehicle data and name."""
        return {
            "vehicle_data": self.vehicle.data,
            "vehicle_name": self.vehicle.name,
        }

    @property
    def unique_id(self) -> str:
        """Return the unique ID."""
        return f"{DOMAIN}-all-data-{self.vehicle.id}"


class DailyDrivingStatsEntity(SensorEntity, HyundaiKiaConnectEntity):
    """Number of days of driving stats, with each day as an attribute."""

    _attr_translation_key = "daily_driving_stats"

    def __init__(
        self,
        coordinator: HyundaiKiaConnectDataUpdateCoordinator,
        vehicle: Vehicle,
    ) -> None:
        """Initialize the daily driving stats sensor."""
        super().__init__(coordinator, vehicle)

    @property
    def native_value(self) -> StateType:
        """Return the number of days with driving stats."""
        return len(self.vehicle.daily_stats)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the driving stats for each day, keyed by date."""
        m: dict[str, Any] = {}
        for day in self.vehicle.daily_stats:
            if day.date is None:
                continue
            key = day.date.strftime("%Y-%m-%d")
            value = {
                "total_consumed": day.total_consumed,
                "engine_consumption": day.engine_consumption,
                "climate_consumption": day.climate_consumption,
                "onboard_electronics_consumption": day.onboard_electronics_consumption,
                "battery_care_consumption": day.battery_care_consumption,
                "regenerated_energy": day.regenerated_energy,
                "distance": day.distance,
            }
            m[key] = value
        return m

    @property
    def unique_id(self) -> str:
        """Return the unique ID."""
        return f"{DOMAIN}-daily-driving-stats-{self.vehicle.id}"

    @property
    def native_unit_of_measurement(self) -> str:
        """Return days as the unit."""
        return UnitOfTime.DAYS


class TodaysDailyDrivingStatsEntity(SensorEntity, HyundaiKiaConnectEntity):
    """Today's driving stats, zeroed until the API reports today."""

    _attr_translation_key = "todays_daily_driving_stats"

    def __init__(
        self,
        coordinator: HyundaiKiaConnectDataUpdateCoordinator,
        vehicle: Vehicle,
    ) -> None:
        """Initialize today's driving stats sensor."""
        super().__init__(coordinator, vehicle)

    @property
    def native_value(self) -> StateType:
        """Return today's date."""
        today = date.today()
        todayskey = today.strftime("%Y-%m-%d")
        return todayskey

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return today's driving stats."""
        today = date.today()
        todayskey = today.strftime("%Y-%m-%d")
        m: dict[str, Any] = {
            "today_date": todayskey,
            "total_consumed": 0,
            "engine_consumption": 0,
            "climate_consumption": 0,
            "onboard_electronics_consumption": 0,
            "battery_care_consumption": 0,
            "regenerated_energy": 0,
            "distance": 0,
        }
        for day in self.vehicle.daily_stats:
            if day.date is None:
                continue
            key = day.date.strftime("%Y-%m-%d")
            if key == todayskey:
                todayvalue = {
                    "today_date": key,
                    "total_consumed": day.total_consumed,
                    "engine_consumption": day.engine_consumption,
                    "climate_consumption": day.climate_consumption,
                    "onboard_electronics_consumption": day.onboard_electronics_consumption,
                    "battery_care_consumption": day.battery_care_consumption,
                    "regenerated_energy": day.regenerated_energy,
                    "distance": day.distance,
                }
                m = todayvalue
                break
        return m

    @property
    def unique_id(self) -> str:
        """Return the unique ID."""
        return f"{DOMAIN}-todays-daily-driving-stats-{self.vehicle.id}"


class SVMStatusSensor(SensorEntity, HyundaiKiaConnectEntity):
    """SVM capture metadata sensor (companion to the SVM image entity).

    The image entity cannot expose extra state attributes (ImageEntity
    finalizes state_attributes), so metadata is exposed here.
    """

    _attr_translation_key = "svm_status"
    _attr_icon = "mdi:camera-iris"
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(
        self,
        coordinator: HyundaiKiaConnectDataUpdateCoordinator,
        vehicle: Vehicle,
    ) -> None:
        """Initialize the SVM status sensor."""
        super().__init__(coordinator, vehicle)
        self._attr_unique_id = f"{DOMAIN}_{vehicle.id}_svm_status"

    @property
    def available(self) -> bool:
        """Return True if cached SVM details are available."""
        return self.coordinator.get_cached_svm_details(self.vehicle.id) is not None

    @property
    def native_value(self) -> datetime | None:
        """Return the capture timestamp of the latest SVM image."""
        details = self.coordinator.get_cached_svm_details(self.vehicle.id)
        if details is None:
            return None
        captured_at: datetime | None = details.captured_at
        return captured_at

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return SVM metadata attributes."""
        details = self.coordinator.get_cached_svm_details(self.vehicle.id)
        if details is None:
            return {}
        return {
            "heading": details.heading,
            "speed": (
                {"value": details.speed[0], "unit": details.speed[1]}
                if details.speed and details.speed[0] is not None
                else None
            ),
            "door_open": details.door_open,
            "trunk_open": details.trunk_open,
        }
