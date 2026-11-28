"""What the records of a DJI app flight log hold (see :mod:`.txtlog` for the container).

Each parser takes a record's decoded data and returns its fields as columns
(``<section>_<field>``), in the units of the table: degrees, metres, m/s,
seconds, volts, amperes, degrees Celsius. Codes (flight mode, flight action...)
are kept as DJI's integers.

==========  ========================  ===========================================
type        section                   notes
==========  ========================  ===========================================
1           ``osd_``                  aircraft state, 10 Hz: starts a frame
2           ``home_``                 home point
3           ``gimbal_``               gimbal attitude
4, 62       ``rc_``                   sticks (62: as the remote controller shows them)
5           ``app_``                  the app's time, distance flown, shooting
9, 10       ``app_tip``, ``app_warn`` the app's messages
14          ``app_gps_``              the app's device position (0.5 Hz)
19          ``mc_``                   flight controller settings
22          ``battery<i>_``           smart batteries
45          ``rtk_``                  RTK solution, 1 Hz (see below)
48          ``osd_abs_``              position with a height of known reference, the aircraft's time
63          ``fc_``                   flight time left, return home and landing estimates
49          ``link_``                 radio link signal (up and down)
15, 40      (details)                 firmware versions, component serial numbers
==========  ========================  ===========================================

The full list of the record types of a Matrice 400 log, those read and those
not, is in the docs (API reference, Drones, Matrice 400 flight record).

Types 45, 48 and 63 follow dji-log-parser 0.6.0 (MIT, L. Vauvillier), checked
on Matrice 400 logs (November 2026); byte 58 of type 48 (the aircraft's
height) was found on those logs:

* 48 (each frame): 0 f64 longitude, 8 f64 latitude (rad); 16 f32 height above
  the take-off point; 20 i16 yaw x 10; 24 u32 seconds + 28 u32 nanoseconds,
  the aircraft's time (UNIX, UTC on these logs); 33 u8 satellites; 34, 42 f64
  home position (rad); 50 f32 height of the take-off point (ellipsoidal);
  54 f32 horizontal distance to the home point; 58 f32 height of the aircraft
  (= 50 + its height above the take-off point); 66 f32 geoid height.
* 45 (1 Hz): 1 u8 RTK solution flag (50: fixed); 2-10 satellites by receiver
  (antenna 1, antenna 2, base) and constellation (GPS, BeiDou, GLONASS); 11 f64
  latitude, 19 f64 longitude (deg), 27 f32 ellipsoidal height; 31, 39 f64, 47
  f32 the position the base broadcasts; 65 3 x f32 standard deviations
  (latitude, longitude, height, m); 77-79 Galileo, 80-82 QZSS satellites.
  Satellite counts are stored shifted by one bit, the low bit set when known.
* 63 (1 Hz, subtype 0): flight time left, time to return home (0xFFFF: none)
  and to land (s), battery needed to return home and to land (%).
"""

from __future__ import annotations

import math
import struct
from collections.abc import Callable
from typing import Any

OSD, HOME, GIMBAL, RC, APP_TIP, APP_WARN, RC_DISPLAY = 1, 2, 3, 4, 9, 10, 62
SMART_BATTERY_GROUP, RTK, POSITION, FC_OSD = 22, 45, 48, 63
CUSTOM, APP_GPS, MC_PARAMS, LINK = 5, 14, 19, 49

Row = dict[str, Any]


def _bit(byte: int, mask: int) -> int:
    """Return the bits of ``byte`` selected by ``mask``, shifted down."""
    byte &= mask
    while mask and not mask & 1:
        byte, mask = byte >> 1, mask >> 1
    return byte


def osd(data: bytes, version: int) -> Row:
    """Return the aircraft's state (type 1)."""
    lon, lat = struct.unpack_from("<dd", data, 0)
    height, vx, vy, vz, pitch, roll, yaw = struct.unpack_from("<7h", data, 16)
    b = data[30:36]  # bit fields (30, 32-35) and the app's command (31)
    (gps_num, flight_action, motor_fail, b6, battery, swave) = struct.unpack_from(
        "<6B", data, 36
    )
    (fly_time,) = struct.unpack_from("<H", data, 42)
    row: Row = {
        "osd_longitude": math.degrees(lon),
        "osd_latitude": math.degrees(lat),
        "osd_height": height / 10.0,  # above the take-off point
        "osd_x_speed": vx / 10.0,
        "osd_y_speed": vy / 10.0,
        "osd_z_speed": vz / 10.0,
        "osd_h_speed": math.hypot(vx, vy) / 10.0,
        "osd_pitch": pitch / 10.0,
        "osd_roll": roll / 10.0,
        "osd_yaw": yaw / 10.0,
        "osd_flight_mode": _bit(b[0], 0x7F),
        "osd_rc_out_of_control": bool(_bit(b[0], 0x80)),
        "osd_app_command": b[1],
        "osd_ground_or_sky": _bit(b[2], 0x06),
        "osd_is_motor_on": bool(_bit(b[2], 0x08)),
        "osd_go_home_status": _bit(b[2], 0xE0),
        "osd_is_vision_used": bool(_bit(b[3], 0x01)),
        "osd_voltage_warning": _bit(b[3], 0x06),
        "osd_is_gps_valid": bool(_bit(b[3], 0x80)),
        "osd_is_compass_error": bool(_bit(b[4], 0x01)),
        "osd_gps_level": _bit(b[4], 0x3C),
        "osd_is_out_of_limit": bool(_bit(b[5], 0x01)),
        "osd_is_motor_blocked": bool(_bit(b[5], 0x08)),
        "osd_is_not_enough_force": bool(_bit(b[5], 0x10)),
        "osd_is_vibrating": bool(_bit(b[5], 0x40)),
        "osd_gps_num": gps_num,
        "osd_flight_action": flight_action,
        "osd_motor_start_failed_cause": motor_fail,
        "osd_non_gps_cause": _bit(b6, 0x0F),
        "osd_battery_percent": battery,
        "osd_vps_height": swave / 10.0,
        "osd_fly_time": fly_time / 10.0,
    }
    if version >= 2 and len(data) > 48:
        row["osd_drone_type"] = data[48]
    if len(data) >= 82:  # Matrice 400 (85 bytes): from the home point, in metres
        row["osd_north"], row["osd_east"] = struct.unpack_from("<ff", data, 74)
    return row


def home(data: bytes, version: int) -> Row:
    """Return the home point (type 2). Its ``altitude`` is the app's (barometric on the Matrice 400)."""
    lon, lat, alt = struct.unpack_from("<ddf", data, 0)
    flags = data[20]
    go_home_height, _lock = struct.unpack_from("<Hh", data, 22)
    (index,) = struct.unpack_from("<H", data, 30)
    lat, lon = math.degrees(lat), math.degrees(lon)
    if not (abs(lat) <= 90 and abs(lon) <= 180):  # no fix yet (a sentinel)
        lat = lon = float("nan")
    row: Row = {
        "home_longitude": lon,
        "home_latitude": lat,
        "home_altitude": alt / 10.0,
        "home_is_home_record": bool(_bit(flags, 0x01)),
        "home_go_home_height": go_home_height,
        "home_flight_record_index": index,
    }
    if version >= 8 and len(data) >= 41:
        (row["home_max_allowed_height"],) = struct.unpack_from("<f", data, 37)
    return row


def gimbal(data: bytes, version: int) -> Row:
    """Return the gimbal's attitude (type 3)."""
    pitch, roll, yaw = struct.unpack_from("<3h", data, 0)
    limits = data[10] if len(data) > 10 else 0
    return {
        "gimbal_pitch": pitch / 10.0,
        "gimbal_roll": roll / 10.0,
        "gimbal_yaw": yaw / 10.0,
        "gimbal_mode": _bit(data[6], 0xC0),
        "gimbal_is_pitch_at_limit": bool(_bit(limits, 0x01)),
        "gimbal_is_roll_at_limit": bool(_bit(limits, 0x02)),
        "gimbal_is_yaw_at_limit": bool(_bit(limits, 0x04)),
        "gimbal_is_stuck": bool(_bit(limits, 0x40)),
    }


def rc(data: bytes, version: int) -> Row:
    """Return the remote controller's sticks (type 4), raw (1024 is the centre)."""
    aileron, elevator, throttle, rudder = struct.unpack_from("<4H", data, 0)
    return {
        "rc_aileron": aileron,
        "rc_elevator": elevator,
        "rc_throttle": throttle,
        "rc_rudder": rudder,
    }


def rc_display(data: bytes, version: int) -> Row:
    """Return the sticks as the remote controller shows them (type 62): after 7 bytes, as type 4."""
    return rc(data[7:], version)


def custom(data: bytes, version: int) -> Row:
    """Return the app's own record (type 5): its time, the distance flown, shooting.

    ``app_time`` is the app's clock (UNIX s, UTC); on Matrice 400 logs it agrees
    with ``start_time + osd_fly_time`` within 0.1 s, and runs 0.24-0.49 s behind
    the aircraft's time (record 48). Its horizontal speed
    (bytes 2-5) is the OSD's and is not kept.
    """
    photo, video, _h_speed, distance_km, ms = struct.unpack_from("<BBffq", data, 0)
    return {
        "app_is_photo": bool(photo),
        "app_is_video": bool(video),
        "app_distance": distance_km * 1000.0,  # flown since take-off, m
        "app_time": ms / 1000.0 if ms > 0 else float("nan"),
    }


def app_gps(data: bytes, version: int) -> Row:
    """Return the position of the app's device (type 14): the remote controller's GNSS."""
    lat, lon, alt = struct.unpack_from("<ddf", data, 0)
    return {"app_gps_latitude": lat, "app_gps_longitude": lon, "app_gps_altitude": alt}


def mc_params(data: bytes, version: int) -> Row:
    """Return the flight controller's settings (type 19).

    ``mc_fail_safe``: the action on a lost link (0 hover, 1 land, 2 go home).
    """
    return {
        "mc_fail_safe": data[0],
        "mc_is_vision_positioning_on": bool(_bit(data[1], 0x01)),
        "mc_is_obstacle_avoidance_on": bool(_bit(data[1], 0x02)),
        "mc_is_user_avoidance_on": bool(_bit(data[1], 0x04)),
    }


def link(data: bytes, version: int) -> Row:
    """Return the radio link's signal (type 49, one byte): up- or downlink, in %."""
    if len(data) != 1:
        return {}  # a longer variant, not read
    side = "up" if data[0] & 0x80 else "down"
    return {f"link_{side}link_signal": data[0] & 0x7F}


def _message(data: bytes) -> str:
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


def app_tip(data: bytes, version: int) -> Row:
    """Return a message of the app (type 9)."""
    return {"app_tip": _message(data)}


def app_warn(data: bytes, version: int) -> Row:
    """Return a warning of the app (type 10)."""
    return {"app_warn": _message(data)}


def smart_battery(data: bytes, version: int) -> Row:
    """One battery's state (type 22): its dynamic part, or its cells' voltages.

    The columns are numbered by the battery's index (``battery1_voltage``...);
    the static part (serial number, design) is not kept.
    """
    kind = data[0]
    if kind == 2 and len(data) >= 22:
        i = data[1]
        mv, ma, full, remaining, temp, cells, percent = struct.unpack_from(
            "<iiIIhBB", data, 2
        )
        return {
            f"battery{i}_voltage": mv / 1000.0,
            f"battery{i}_current": ma / 1000.0,  # negative: discharging
            f"battery{i}_full_capacity": full,
            f"battery{i}_remaining_capacity": remaining,
            f"battery{i}_temperature": temp / 10.0,
            f"battery{i}_percent": percent,
        }
    if kind == 3 and len(data) >= 3:
        i, n = data[1], data[2]
        n = min(n, (len(data) - 3) // 2)
        volts = struct.unpack_from(f"<{n}H", data, 3)
        return {
            f"battery{i}_cell_voltage_{k + 1}": v / 1000.0 for k, v in enumerate(volts)
        }
    return {}


def position(data: bytes, version: int) -> Row:
    """Return the aircraft's position and time (type 48, see the notes).

    Layout as dji-log-parser 0.6.0 (``GroundStationPushData``), plus the
    aircraft's height at byte 58 (found on Matrice 400 logs).
    """
    lon, lat = struct.unpack_from("<dd", data, 0)
    (yaw,) = struct.unpack_from("<h", data, 20)
    seconds, nanoseconds = struct.unpack_from("<II", data, 24)
    home_h, home_distance, h = struct.unpack_from("<fff", data, 50)
    row: Row = {
        "osd_aircraft_time": seconds + nanoseconds * 1e-9,
        "osd_abs_latitude": math.degrees(lat),
        "osd_abs_longitude": math.degrees(lon),
        "osd_abs_altitude": h,
        "osd_abs_yaw": yaw / 10.0,
        "osd_abs_gps_num": data[33],
        "osd_home_distance": home_distance,
        "home_abs_altitude": home_h,
    }
    if len(data) >= 70:
        (row["osd_geoid_height"],) = struct.unpack_from("<f", data, 66)
    return row


def _satellites(value: int) -> int | None:
    """Return a satellite count: stored shifted by one bit, the low bit set when known."""
    return value >> 1 if value & 0x01 else None


RTK_RECEIVERS = ("antenna1", "antenna2", "base")
RTK_CONSTELLATIONS = ("gps", "beidou", "glonass")


def rtk(data: bytes, version: int) -> Row:
    """Return the RTK solution (type 45, see the notes).

    Layout as dji-log-parser 0.6.0 (``RTKDifference``). ``rtk_pos_flag``:
    0-15 none, 16 single, 17-49 float, 50 fixed.
    """
    lat, lon, h = struct.unpack_from("<ddf", data, 11)
    base_lat, base_lon, base_h = struct.unpack_from("<ddf", data, 31)
    row: Row = {
        "rtk_pos_flag": data[1],
        "rtk_latitude": lat,
        "rtk_longitude": lon,
        "rtk_height": h,
        "rtk_base_latitude": base_lat,
        "rtk_base_longitude": base_lon,
        "rtk_base_height": base_h,
    }
    if (base_lat, base_lon) == (0.0, 0.0):  # no base station
        row["rtk_base_latitude"] = row["rtk_base_longitude"] = None
    if base_h == 0.0:
        row["rtk_base_height"] = None
    if len(data) >= 77:
        std = struct.unpack_from("<3f", data, 65)
        row["rtk_std_latitude"], row["rtk_std_longitude"], row["rtk_std_height"] = std
    for r, receiver in enumerate(RTK_RECEIVERS):
        for c, constellation in enumerate(RTK_CONSTELLATIONS):
            row[f"rtk_{receiver}_{constellation}"] = _satellites(data[2 + 3 * r + c])
        if len(data) >= 83:
            row[f"rtk_{receiver}_galileo"] = _satellites(data[77 + r])
            row[f"rtk_{receiver}_qzss"] = _satellites(data[80 + r])
    return row


def fc_osd(data: bytes, version: int) -> Row:
    """Return the flight controller's battery estimates (type 63, subtype 0).

    Layout as dji-log-parser 0.6.0 (``FlightControllerCommonOSDField``,
    ``BatteryOsd``): flight time left, time and battery needed to return home
    and to land. Other subtypes (1 counts down from 9, unknown) are not read.
    """
    (subtype,) = struct.unpack_from("<H", data, 0)
    if subtype != 0 or len(data) < 10:
        return {}
    useful, go_home, land, go_home_battery, land_battery = struct.unpack_from(
        "<HHHBB", data, 2
    )
    return {
        "fc_flight_time_left": useful,  # s, 0 on the ground
        "fc_go_home_time": go_home if go_home != 0xFFFF else None,  # s
        "fc_land_time": land,  # s
        "fc_go_home_battery": go_home_battery,  # % needed to return home
        "fc_land_battery": land_battery,  # % needed to land
    }


FIRMWARE, COMPONENT_SERIAL = 15, 40
SENDERS = {
    0: "none",
    1: "camera",
    3: "flight controller",
    4: "gimbal",
    6: "remote controller",
    11: "battery",
}
COMPONENTS = {1: "camera", 2: "aircraft", 3: "remote controller", 4: "battery"}


def metadata(records: Any) -> dict[str, Any]:
    """Return what the log says of its components: firmware versions (type 15), serial numbers (type 40).

    Parameters
    ----------
    records : iterable of txtlog.Record
    """
    firmware: dict[str, str] = {}
    serials: dict[str, str] = {}
    for record in records:
        data = record.data
        if record.type == FIRMWARE and len(data) >= 6:
            who = SENDERS.get(data[0], f"sender {data[0]}")
            if data[1]:
                who += f" {data[1]}"
            firmware[who] = ".".join(str(b) for b in data[2:6])
        elif record.type == COMPONENT_SERIAL and len(data) >= 3:
            (kind,) = struct.unpack_from("<H", data, 0)
            n = data[2]
            serials[COMPONENTS.get(kind, f"component {kind}")] = _message(
                data[3 : 3 + n]
            )
    return {"firmware": firmware, "component_serials": serials}


# type -> (parser, minimum size, whether its values hold until the next record)
PARSERS: dict[int, tuple[Callable[[bytes, int], Row], int, bool]] = {
    OSD: (osd, 44, True),
    HOME: (home, 32, True),
    GIMBAL: (gimbal, 7, True),
    RC: (rc, 8, True),
    RC_DISPLAY: (rc_display, 15, True),
    APP_TIP: (app_tip, 0, False),
    APP_WARN: (app_warn, 0, False),
    CUSTOM: (custom, 18, False),
    APP_GPS: (app_gps, 20, True),
    MC_PARAMS: (mc_params, 2, True),
    LINK: (link, 1, True),
    SMART_BATTERY_GROUP: (smart_battery, 1, True),
    POSITION: (position, 62, False),
    FC_OSD: (fc_osd, 2, True),
    RTK: (rtk, 51, False),
}
"""The records read: an event (app message) or a value of its own epoch
(RTK, record 48, the app's time) only in its frame, a state (OSD, home...)
until it changes."""


def parse(kind: int, data: bytes, version: int) -> tuple[Row, bool] | None:
    """Return the columns of a record and whether they hold; None for a record not read or cut short."""
    if kind not in PARSERS:
        return None
    parser, size, holds = PARSERS[kind]
    if len(data) < size:
        return None
    try:
        return parser(data, version), holds
    except (struct.error, IndexError):
        return None
