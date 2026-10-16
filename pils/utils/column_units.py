"""Units of the columns of the flight tables.

The units are stored with each table in HDF5 (``units`` attribute, JSON) and
can be looked up with :func:`column_unit`. Columns whose unit is unknown or
that have none (counts, flags, quaternions) are not listed.
"""

import re

# By full column name, or by the part after the last ":"
_EXACT = {
    # Time base of every table
    "timestamp": "s",
    "timestamp_us": "us",
    "timestamp_ns": "ns",
    "monotonic_ns": "ns",
    "unix_time_ms": "ms",
    "tick": "1/4.5 MHz",
    # Standardized coordinates of the synchronized GPS tables
    "latitude": "deg",
    "longitude": "deg",
    "altitude": "m",
    # Payload GPS (u-blox units, height converted to m by the loader)
    "posllh_lat": "deg",
    "posllh_lon": "deg",
    "posllh_height": "m",
    "posllh_hMSL": "mm",
    "posllh_hAcc": "mm",
    "posllh_vAcc": "mm",
    "posecef_ecefX": "cm",
    "posecef_ecefY": "cm",
    "posecef_ecefZ": "cm",
    "posecef_pAcc": "cm",
    "clock_clkB": "ns",
    "clock_clkD": "ns/s",
    "clock_tAcc": "ns",
    "clock_fAcc": "ps/s",
    # ADC (ADS1015)
    "amplitude": "mV",
    # Payload IMU
    "acc_x": "g",
    "acc_y": "g",
    "acc_z": "g",
    "gyro_x": "rad/s",
    "gyro_y": "rad/s",
    "gyro_z": "rad/s",
    "pressure": "hPa",
    "temperature": "degC",
    # Inclinometer
    "roll": "deg",
    "pitch": "deg",
    "yaw": "deg",
    # Litchi
    "gimbalPitch": "deg",
    "gimbalRoll": "deg",
    "gimbalYaw": "deg",
    "gimbalYawAbs": "deg",
    "pitchRaw": "0.1 deg",
    "rollRaw": "0.1 deg",
    "yawRaw": "0.1 deg",
    "gimbalPitchRaw": "0.1 deg",
    "gimbalRollRaw": "0.1 deg",
    "gimbalYawRaw": "0.1 deg",
    # AZEL
    "az": "deg",
    "el": "deg",
    "srange": "m",
    "E": "m",
    "N": "m",
    "U": "m",
}

# DJI drone table and .dat records: (pattern on the full name, unit)
_PATTERNS = [
    (r"^GPS(\(\d\))?:(Latitude|Longitude|latitude|longitude)$", "deg"),
    (r"^GPS(\(\d\))?:heightMSL$", "m"),
    (r"^GPS(\(\d\))?:vel[NED]$", "m/s"),
    (r"^RTK:(lat|lon)_[ps]$", "deg"),
    (r"^RTK:hmsl_[ps]$", "m"),
    (r"^RTK:vel_[ned]$", "m/s"),
    (r"^RTK:(yaw|pitch)$", "deg"),
    (r"^IMU_ATTI\(\d\):(latitude|longitude|roll|pitch|yaw)$", "deg"),
    (r"^IMU_ATTI\(\d\):(press|alti)$", "m"),
    (r"^IMU_ATTI\(\d\):accel[XYZ]$", "g"),
    (r"^IMU_ATTI\(\d\):gyro[XYZ]$", "deg/s"),
    (r"^IMU_ATTI\(\d\):vel[NED]$", "m/s"),
    (r"^IMU_ATTI\(\d\):temperature$", "degC"),
    (r"^osd_data:(relativeHeight|vpsHeight)$", "m"),
    (r"^osd_data:flightTime$", "ms"),
    (r"^BatteryInfo:(ad_v|pack_ve)$", "V"),
    (r"^BatteryInfo:(ave_I|I)$", "A"),
    (r"^BatteryInfo:r_time$", "s"),
    (r"^BatteryInfo:cap_per$", "%"),
    (r"^BatteryInfo:temp$", "degC"),
    (r"^AirSpeed:windSpeed$", "m/s"),
    (r"^MotorCtrl:PWM:", "%"),
    (r"^Motor:Current:", "A"),
    (r"^Motor:Speed:", "rpm"),
    (r"^Motor:EscTemp:", "degC"),
    (r"^Motor:PPM(recv|send):", "%"),
]

# Units in the name (Litchi, inclinometer, LM76): only for names without ":",
# so that DJI fields such as RTK:svn_s (a count) are not matched
_SUFFIXES = [
    (r"\(deg\)$", "deg"),
    (r"\(m\)$", "m"),
    (r"\(mps\)$", "m/s"),
    # Inclinometer and LM76 units in the name
    (r"_deg_s$", "deg/s"),
    (r"_rad_s$", "rad/s"),
    (r"_m_s2$", "m/s^2"),
    (r"_m_s$", "m/s"),
    (r"_deg$", "deg"),
    (r"_rad$", "rad"),
    (r"_m$", "m"),
    (r"_s$", "s"),
    (r"_c$", "degC"),
]
_COMPILED = [(re.compile(pattern), unit) for pattern, unit in _PATTERNS]
_COMPILED_SUFFIXES = [(re.compile(pattern), unit) for pattern, unit in _SUFFIXES]


def column_unit(column: str) -> str | None:
    """Unit of a column, or None if it has none or it is not known.

    Examples
    --------
    >>> column_unit("RTK:lat_p")
    'deg'
    >>> column_unit("pqr_P_rad_s")
    'rad/s'
    >>> column_unit("Motor:Speed:RFront")
    'rpm'
    """
    if column in _EXACT:
        return _EXACT[column]
    for pattern, unit in _COMPILED:
        if pattern.search(column):
            return unit
    if ":" not in column:
        for pattern, unit in _COMPILED_SUFFIXES:
            if pattern.search(column):
                return unit
    return None


def table_units(columns: list[str]) -> dict[str, str]:
    """Units of the columns that have a known unit."""
    units = {}
    for column in columns:
        unit = column_unit(column)
        if unit is not None:
            units[column] = unit
    return units
