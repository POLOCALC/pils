"""Column names, units and conversions of the flight tables (schema v1).

The loaders (:mod:`pils.sensors`, :mod:`pils.drones`) read each file with the
names of its format. :class:`pils.flight.Flight` renames the columns to the
names below, converts their units where needed, and drops duplicates, so
that every table follows one convention:

- ``snake_case``;
- the unit as a suffix (``_deg``, ``_m``, ``_m_s``, ``_rad_s``, ``_g``,
  ``_hpa``, ``_c``, ``_mv``, ``_pct``, ...); no suffix for counts, flags,
  states, quaternions and values whose unit is not known;
- heights say their reference: ``h_ell_m`` (above the WGS84 ellipsoid),
  ``h_msl_m`` (above mean sea level), ``h_rel_m`` (above take-off), ``h_m``
  (unknown);
- ``timestamp`` (UNIX seconds, UTC) is the time base of every table;
  ``datetime`` is the same instant as ``Datetime("us", "UTC")``.

Columns not listed here are converted to ``snake_case`` by
:func:`snake_case`. :func:`unit_of` gives the unit of a column from its
name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Field:
    """A renamed column: its new name and the factor to its unit."""

    name: str
    scale: float | None = None


# Columns with the same name everywhere
_COMMON = {
    "timestamp": Field("timestamp"),
    "datetime": Field("datetime"),
    "tick": Field("tick"),
    "msg_type": Field("msg_type"),
    "unix_time_ms": Field("unix_time_ms"),
    "timestamp_us": Field("timestamp_us"),
    "timestamp_ns": Field("timestamp_ns"),
    "monotonic_ns": Field("monotonic_ns"),
    "datetime_relative": Field("datetime_relative"),
}


def _xyz(native: str, name: str, unit: str, axes: str = "xyz") -> dict[str, Field]:
    """``{native}{X}`` -> ``{name}_{x}_{unit}`` for each axis."""
    return {
        native.format(axis=a): Field(f"{name}_{b}" + (f"_{unit}" if unit else ""))
        for a, b in zip(axes, "xyz", strict=True)
    }


PAYLOAD_GPS = {
    "posllh_lat": Field("lat_deg"),
    "posllh_lon": Field("lon_deg"),
    "posllh_height": Field("h_ell_m"),
    "posllh_hMSL": Field("h_msl_m", 1e-3),  # mm
    "posllh_hAcc": Field("h_acc_m", 1e-3),  # mm
    "posllh_vAcc": Field("v_acc_m", 1e-3),  # mm
    "posllh_msgmode": Field("posllh_msg_mode"),
    "posecef_ecefX": Field("ecef_x_m", 1e-2),  # cm
    "posecef_ecefY": Field("ecef_y_m", 1e-2),
    "posecef_ecefZ": Field("ecef_z_m", 1e-2),
    "posecef_pAcc": Field("p_acc_m", 1e-2),  # cm
    "posecef_msgmode": Field("posecef_msg_mode"),
    "clock_clkB": Field("clock_bias_ns"),
    "clock_clkD": Field("clock_drift_ns_s"),
    "clock_tAcc": Field("time_acc_ns"),
    "clock_fAcc": Field("freq_acc_ps_s"),
    "clock_msgmode": Field("clock_msg_mode"),
}

ADC = {
    "amplitude": Field("amplitude_mv"),
}

LM76 = {
    "temperature_c": Field("temperature_c"),
}

PAYLOAD_IMU = {
    "pressure": Field("pressure_hpa"),
    "temperature": Field("temperature_c"),
    **_xyz("acc_{axis}", "acc", "g"),
    **_xyz("gyro_{axis}", "gyro", "rad_s"),
    **_xyz("mag_{axis}", "mag", ""),
}

INCLINOMETER = {
    # IMX-5 (INS, IMU, INL2 tables)
    "lat_deg": Field("lat_deg"),
    "lon_deg": Field("lon_deg"),
    "alt_m": Field("h_m"),  # not referenced to GPS without a fix
    "roll": Field("roll_deg"),
    "pitch": Field("pitch_deg"),
    "yaw": Field("yaw_deg"),
    # Kernel-100
    "Temper": Field("temperature_c"),
}

LITCHI = {
    "latitude": Field("lat_deg"),
    "longitude": Field("lon_deg"),
    "altitude(m)": Field("h_rel_m"),
    "speed(mps)": Field("speed_m_s"),
    "distance(m)": Field("distance_m"),
    "velocityX(mps)": Field("vel_x_m_s"),
    "velocityY(mps)": Field("vel_y_m_s"),
    "velocityZ(mps)": Field("vel_z_m_s"),
    "pitch(deg)": Field("pitch_deg"),
    "roll(deg)": Field("roll_deg"),
    "yaw(deg)": Field("yaw_deg"),
    "isflying": Field("is_flying"),
    "batteryTemperature": Field("battery_temperature"),
    "gimbalPitch": Field("gimbal_pitch_deg"),
    "gimbalRoll": Field("gimbal_roll_deg"),
    "gimbalYaw": Field("gimbal_yaw_deg"),  # relative to the aircraft
    "gimbalYawAbs": Field("gimbal_heading_deg"),  # from north
}
# 0.1 deg duplicates of the angles above
LITCHI_DROP = (
    "pitchRaw",
    "rollRaw",
    "yawRaw",
    "gimbalPitchRaw",
    "gimbalRollRaw",
    "gimbalYawRaw",
)


def _gnss(prefix: str) -> dict[str, Field]:
    """DJI GNSS receiver fields (drone table and GPS(n) records)."""
    p = f"{prefix}_" if prefix else ""
    return {
        "date": Field(f"{p}date"),
        "time": Field(f"{p}time"),
        "latitude": Field(f"{p}lat_deg"),
        "longitude": Field(f"{p}lon_deg"),
        "heightMSL": Field(f"{p}h_msl_m"),
        "velN": Field(f"{p}vel_n_m_s"),
        "velE": Field(f"{p}vel_e_m_s"),
        "velD": Field(f"{p}vel_d_m_s"),
        "hdop": Field(f"{p}hdop"),
        "pdop": Field(f"{p}pdop"),
        "hacc": Field(f"{p}h_acc"),
        "sacc": Field(f"{p}s_acc"),
        "numGPS": Field(f"{p}num_gps"),
        "numGLN": Field(f"{p}num_glonass"),
        "numSV": Field(f"{p}num_sv"),
    }


DJI = {
    **{f"GPS:{k}": v for k, v in _gnss("gps").items()},
    "RTK:date": Field("rtk_date"),
    "RTK:time": Field("rtk_time"),
    "RTK:lat_p": Field("rtk_lat_deg"),
    "RTK:lon_p": Field("rtk_lon_deg"),
    # Named hmsl by DJI, but referenced to the base position, which the
    # POLOCALC bases broadcast as ellipsoidal heights
    "RTK:hmsl_p": Field("rtk_h_ell_m"),
    "RTK:lat_s": Field("rtk_lat2_deg"),
    "RTK:lon_s": Field("rtk_lon2_deg"),
    "RTK:hmsl_s": Field("rtk_h2_m"),
    "RTK:vel_n": Field("rtk_vel_n_m_s"),
    "RTK:vel_e": Field("rtk_vel_e_m_s"),
    "RTK:vel_d": Field("rtk_vel_d_m_s"),
    "RTK:yaw": Field("rtk_yaw_deg"),  # antenna baseline, IMU yaw + 90 deg
    "RTK:pitch": Field("rtk_pitch_deg"),
    "RTK:svn_p": Field("rtk_num_sv"),
    "RTK:svn_s": Field("rtk_num_sv2"),
    "RTK:hdop": Field("rtk_hdop"),
    **{f"RTK:pos_flg_{i}": Field(f"rtk_pos_flag_{i}") for i in range(6)},
    "RTK:gps_state": Field("rtk_gps_state"),
    # DatCon CSV
    "GPS:Lat[degrees]": Field("gps_lat_deg"),
    "GPS:Long[degrees]": Field("gps_lon_deg"),
    "GPS:heightMSL[meters]": Field("gps_h_msl_m"),
    "GPS:Date": Field("gps_date"),
    "GPS:Time": Field("gps_time"),
    "RTKdata:Lat_P": Field("rtk_lat_deg"),
    "RTKdata:Lon_P": Field("rtk_lon_deg"),
    "RTKdata:Hmsl_P": Field("rtk_h_ell_m"),
}

_MOTORS = ("RFront", "LFront", "LSide", "LBack", "RBack", "RSide")

# DJI .dat records: fields after the "<record>:" prefix
DJI_RECORDS = {
    "IMU_ATTI": {
        "longitude": Field("lon_deg"),
        "latitude": Field("lat_deg"),
        "press": Field("h_baro_m"),  # barometric altitude
        "alti": Field("h_fused_m"),  # fused altitude
        **_xyz("accel{axis}", "acc", "g", "XYZ"),
        **_xyz("gyro{axis}", "gyro", "deg_s", "XYZ"),
        **_xyz("ag{axis}", "ag", "", "XYZ"),
        **_xyz("gb{axis}", "gyro_bias", "", "XYZ"),
        **_xyz("mag{axis}", "mag", "", "XYZ"),
        "quatW": Field("quat_w"),
        "quatX": Field("quat_x"),
        "quatY": Field("quat_y"),
        "quatZ": Field("quat_z"),
        "velN": Field("vel_n_m_s"),
        "velE": Field("vel_e_m_s"),
        "velD": Field("vel_d_m_s"),
        "temperature": Field("temperature_c"),
        "numSats": Field("num_sats"),
        "roll": Field("roll_deg"),
        "pitch": Field("pitch_deg"),
        "yaw": Field("yaw_deg"),
    },
    "GPS": _gnss(""),
    "osd_data": {
        "relativeHeight": Field("h_rel_m"),
        "vpsHeight": Field("h_vps_m"),
        "flightTime": Field("flight_time_ms"),
    },
    "BatteryInfo": {
        "ad_v": Field("adc_voltage_v"),
        "r_time": Field("remaining_time_s"),
        "ave_I": Field("current_avg_a"),
        "pack_ve": Field("pack_voltage_v"),
        "I": Field("current_a"),
        "r_cap": Field("remaining_capacity"),
        "cap_per": Field("charge_pct"),
        "temp": Field("temperature_c"),
        "f_cap": Field("full_capacity"),
    },
    "AirSpeed": {
        "windSpeed": Field("wind_speed_m_s"),
        "windX": Field("wind_x"),
        "windY": Field("wind_y"),
        "MotorSpd": Field("motor_speed"),
    },
    "MotorCtrl": {f"PWM:{m}": Field(f"pwm_{m.lower()}_pct") for m in _MOTORS},
    "Motor": {
        native: Field(name)
        for m in _MOTORS
        for native, name in (
            (f"Status:{m}", f"status_{m.lower()}"),
            (f"Current:{m}", f"current_{m.lower()}_a"),
            (f"Speed:{m}", f"speed_{m.lower()}_rpm"),
            (f"Volts:{m}", f"voltage_{m.lower()}_v"),
            (f"EscTemp:{m}", f"esc_temperature_{m.lower()}_c"),
            (f"PPMrecv:{m}", f"ppm_recv_{m.lower()}_pct"),
            (f"V_out:{m}", f"v_out_{m.lower()}"),
            (f"PPMsend:{m}", f"ppm_send_{m.lower()}_pct"),
        )
    },
}

# ArduPilot (BlackSquare) GPS message
BLACKSQUARE = {
    "Lat": Field("lat_deg"),
    "Lng": Field("lon_deg"),
    "Alt": Field("h_msl_m"),
    "Latitude": Field("lat_deg"),
    "Longitude": Field("lon_deg"),
    "heightMSL": Field("h_msl_m"),
}

CAMERA = {
    "pitch": Field("pitch_deg"),
    "roll": Field("roll_deg"),
    "yaw": Field("yaw_deg"),
    "qw": Field("quat_w"),
    "qx": Field("quat_x"),
    "qy": Field("quat_y"),
    "qz": Field("quat_z"),
}

TABLES: dict[str, dict[str, Field]] = {
    "gps": PAYLOAD_GPS,
    "adc": ADC,
    "lm76": LM76,
    "imu": PAYLOAD_IMU,
    "inclinometer": INCLINOMETER,
    "litchi": LITCHI,
    "dji": DJI,
    "blacksquare": BLACKSQUARE,
    "camera": CAMERA,
}
DROP: dict[str, tuple[str, ...]] = {"litchi": LITCHI_DROP}

# Unit of a column from its suffix, longest first
_UNITS = (
    ("_m_s2", "m/s^2"),
    ("_deg_s", "deg/s"),
    ("_rad_s", "rad/s"),
    ("_ns_s", "ns/s"),
    ("_ps_s", "ps/s"),
    ("_m_s", "m/s"),
    ("_deg", "deg"),
    ("_rad", "rad"),
    ("_hpa", "hPa"),
    ("_rpm", "rpm"),
    ("_pct", "%"),
    ("_dbm", "dBm"),
    ("_db", "dB"),
    ("_pw", "pW"),
    ("_mw", "mW"),
    ("_mv", "mV"),
    ("_ms", "ms"),
    ("_us", "us"),
    ("_ns", "ns"),
    ("_m", "m"),
    ("_s", "s"),
    ("_g", "g"),
    ("_c", "degC"),
    ("_v", "V"),
    ("_a", "A"),
)


def unit_of(column: str) -> str | None:
    """Unit of a column, from its name.

    Examples
    --------
    >>> unit_of("rtk_lat_deg"), unit_of("acc_x_m_s2"), unit_of("timestamp")
    ('deg', 'm/s^2', 's')
    >>> unit_of("ins_status") is None
    True
    """
    if column == "timestamp":
        return "s"
    for suffix, unit in _UNITS:
        if column.endswith(suffix):
            return unit
    return None


def table_units(columns: list[str]) -> dict[str, str]:
    """Units of the columns that have one."""
    return {c: u for c in columns if (u := unit_of(c)) is not None}


def snake_case(name: str) -> str:
    """Generic ``snake_case`` for a column not in the schema.

    Examples
    --------
    >>> snake_case("timeOfWeek_s"), snake_case("pqr_P_rad_s")
    ('time_of_week_s', 'pqr_p_rad_s')
    >>> snake_case("Clock:offsetTime"), snake_case("GPS:Lat[degrees]")
    ('clock_offset_time', 'gps_lat_degrees')
    """
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name)
    return name.strip("_").lower()


def record_name(record: str) -> str:
    """Table name of a DJI .dat record, e.g. ``IMU_ATTI(0)`` -> ``imu_atti_0``."""
    return snake_case(record.replace("osd_data", "osd"))


def _record_fields(record: str) -> dict[str, Field]:
    base = re.sub(r"\(\d+\)$", "", record)
    return DJI_RECORDS.get(base, {})


def canonical(
    df: pl.DataFrame, kind: str | None = None, record: str | None = None
) -> pl.DataFrame:
    """A table with the schema's column names and units.

    Parameters
    ----------
    df : pl.DataFrame
        Table as a loader returns it.
    kind : str, optional
        Source: one of :data:`TABLES` (``"gps"``, ``"litchi"``, ``"dji"``,
        ...). None for the generic conversion only.
    record : str, optional
        For a DJI .dat record table: the record name, e.g. ``"IMU_ATTI(0)"``.
        Its columns are ``"<record>:<field>"``.

    Returns
    -------
    pl.DataFrame
        The renamed and converted table. Tables already in the schema are
        returned unchanged.
    """
    fields = _record_fields(record) if record else TABLES.get(kind or "", {})
    drop = set(DROP.get(kind or "", ()))
    prefix = f"{record}:" if record else ""
    exprs = []
    names: dict[str, str] = {}
    for column in df.columns:
        if column in drop:
            continue
        native = (
            column[len(prefix) :] if prefix and column.startswith(prefix) else column
        )
        field = _COMMON.get(column) or fields.get(native) or fields.get(column)
        if field is None:
            field = Field(snake_case(native))
        if field.name in names.values():
            raise ValueError(
                f"Columns {names_of(names, field.name)} and {column} both map to "
                f"{field.name}"
            )
        names[column] = field.name
        expr = pl.col(column)
        if field.scale is not None:
            expr = expr.cast(pl.Float64) * field.scale
        exprs.append(expr.alias(field.name))
    return df.select(exprs)


def names_of(names: dict[str, str], target: str) -> list[str]:
    """Source columns mapped to ``target``."""
    return [k for k, v in names.items() if v == target]


_TABLE_TITLES = {
    "gps": "Payload GPS (`payload_data.gps`)",
    "adc": "ADC (`payload_data.adc`)",
    "lm76": "LM76 (`payload_data.lm76`)",
    "imu": "Payload IMU (`payload_data.imu[...]`)",
    "inclinometer": "Inclinometer (`payload_data.inclinometer[...]`)",
    "litchi": "Litchi (`drone_data.litchi`)",
    "dji": "DJI drone table (`drone_data.drone`)",
    "blacksquare": "BlackSquare / ArduPilot (`drone_data.drone[...]`)",
    "camera": "Camera (`payload_data.camera`)",
}


def markdown_reference() -> str:
    """The renamed columns of every source, as Markdown tables.

    ``docs/user-guide/data-schema.md`` includes this text; a test checks
    that it is up to date.
    """
    lines = []

    def table(title: str, fields: dict[str, Field], drop: tuple[str, ...] = ()):
        lines.extend(
            [f"### {title}", "", "| Source column | Column | Unit |", "|---|---|---|"]
        )
        for native, field in fields.items():
            scale = f" (×{field.scale:g})" if field.scale is not None else ""
            unit = unit_of(field.name) or ""
            lines.append(f"| `{native}` | `{field.name}` | {unit}{scale} |")
        for native in drop:
            lines.append(f"| `{native}` | dropped (duplicate) | |")
        lines.append("")

    for kind, fields in TABLES.items():
        table(_TABLE_TITLES[kind], fields, DROP.get(kind, ()))
    for record, fields in DJI_RECORDS.items():
        name = record_name(record) + ("_<n>" if record in ("IMU_ATTI", "GPS") else "")
        table(f"DJI record `{record}` (`drone_data.records['{name}']`)", fields)
    return "\n".join(lines)
