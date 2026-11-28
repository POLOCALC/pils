"""Decoders of the Matrice 400 telemetry topics (Payload SDK subscriptions).

Ported (copied and adapted, not imported) from LAGER, POLOCALC/lager at 825164a:
``lager/UAVs/DJI_M400/utils.py``; ``decode`` counts what it skips in a dict the
caller gives instead of a module-level counter.
"""

import math
import struct

from . import parameters as params

BATTERY = struct.Struct(
    "<BBiiIIhBB8sBBB"
)  # T_DjiFcSubscriptionSingleBatteryInfo, 33 bytes
ESC = struct.Struct(
    "<hhHhH"
)  # ESCStatusIndividual, 10 bytes (x8 in T_DjiFcSubscriptionEscData)
ESC_COUNT = 8
ANOMALY_FLAGS = (
    "impact_in_air",
    "random_fly",
    "height_ctrl_fail",
    "tilt_ctrl_fail",
    "yaw_ctrl_fail",
    "falling",
    "wind_level1",
    "wind_level2",
    "compass_install_error",
    "imu_install_error",
    "esc_temp_high",
    "esc_disconnected",
    "gps_yaw_error",
)


def quat_to_euler(q0: float, q1: float, q2: float, q3: float) -> tuple:
    """Quaternion (q0=w) -> (roll, pitch, yaw) in degrees, same as DJI_M600.utils."""
    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    sinp = 2 * (q0 * q2 - q3 * q1)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _quaternion(p):
    q0, q1, q2, q3 = struct.unpack("<4f", p)
    roll, pitch, yaw = quat_to_euler(q0, q1, q2, q3)
    return {"q": [q0, q1, q2, q3], "roll": roll, "pitch": pitch, "yaw": yaw}


def _velocity(p):
    vx, vy, vz, info = struct.unpack("<3fB", p)  # m/s, North-East-Up
    return {"vx": vx, "vy": vy, "vz": vz, "vel_info": info}


def _gps_position(p):
    lon, lat, alt = struct.unpack("<3i", p)  # x = lon, y = lat (deg*1e7), z = alt (mm)
    return {
        "gps_lon_deg": lon * 1e-7,
        "gps_lat_deg": lat * 1e-7,
        "gps_hfsl_m": alt / 1000.0,
    }


def _rtk_position(p):
    lon, lat, hfsl = struct.unpack("<ddf", p)  # deg, deg, m
    return {"rtk_lon_deg": lon, "rtk_lat_deg": lat, "rtk_hfsl_m": hfsl}


def _gps_details(p):
    hdop, pdop, fix, vacc, hacc, sacc, n_gps, n_glonass, n_total, counter = (
        struct.unpack("<6f2I2H", p)
    )
    return {
        "hdop": hdop * params.DOP_SCALE,
        "pdop": pdop * params.DOP_SCALE,
        "gps_fix": int(fix) if math.isfinite(fix) else None,
        "gps_vacc_mm": vacc,
        "gps_hacc_mm": hacc,
        "gps_sacc_cms": sacc,
        "gps_nsv": n_total,
        "gps_nsv_gps": n_gps,
        "gps_nsv_glonass": n_glonass,
    }


def _flight_anomaly(p):
    (bits,) = struct.unpack("<I", p)
    out = {"flight_error": bits}
    out.update(
        {f"anomaly_{name}": bool(bits >> b & 1) for b, name in enumerate(ANOMALY_FLAGS)}
    )
    return out


def _battery(prefix):
    def decode(p):
        (_, index, mv, ma, full, remained, temp, cells, percent, state, _, _, sop) = (
            BATTERY.unpack(p)
        )
        bits = int.from_bytes(state[:4], "little")
        return {
            f"{prefix}_index": index,
            f"{prefix}_voltage_mv": mv,
            f"{prefix}_current_ma": ma,
            f"{prefix}_capacity_mah": full,
            f"{prefix}_remaining_mah": remained,
            f"{prefix}_temp_c": temp / 10.0,
            f"{prefix}_cells": cells,
            f"{prefix}_percentage": percent,
            f"{prefix}_sop": sop,
            f"{prefix}_cell_break": (bits >> 12) & 0x1F,
            f"{prefix}_self_check": (bits >> 17) & 0x7,
            f"{prefix}_closed_reason": (bits >> 27) & 0x1F,
        }

    return decode


def _esc(p):
    out = {}
    for i in range(ESC_COUNT):
        ma, rpm, mv, temp, flags = ESC.unpack_from(p, i * ESC.size)
        if not (ma or rpm or mv or temp):
            continue  # slot not used by this aircraft
        n = i + 1
        out.update(
            {
                f"esc{n}_rpm": rpm,
                f"esc{n}_current_a": ma / 1000.0,
                f"esc{n}_voltage_v": mv / 1000.0,
                f"esc{n}_temp_c": temp / 10.0,
                # bits: stall, no_load, unbalanced, disconnected, temp_high
                f"esc{n}_flags": flags,
            }
        )  # 0.1 C on the Matrice 400
    if out:
        out["esc_max_temp_c"] = max(v for k, v in out.items() if k.endswith("_temp_c"))
    return out


def _flat(fmt, names, scales=None):
    """Decode a flat struct: unpack fmt and name the values; scales (optional) multiplies each one."""
    st = struct.Struct(fmt)

    def decode(p):
        values = st.unpack(p)
        if scales:
            values = [v * k for v, k in zip(values, scales, strict=True)]
        return dict(zip(names, values, strict=True))

    return st.size, decode


def _bits(prefix, size, flags):
    """Decode a bit-field struct: the raw value plus one key per named bit (LSB first)."""

    def decode(p):
        bits = int.from_bytes(p, "little")
        out = {prefix: bits}
        out.update(
            {
                f"{prefix}_{name}": bool(bits >> b & 1)
                for b, name in enumerate(flags)
                if name
            }
        )
        return out

    return size, decode


def _acceleration_body(p):
    x, y, z = struct.unpack(
        "<3f", p
    )  # g, gravity included (z ~ -1 at rest) on the Matrice 400
    return {
        "acc_body_x": x,
        "acc_body_y": y,
        "acc_body_z": z,
        "ax": x,
        "ay": y,
        "az": z,
    }


def _position_fused(p):
    # 24 bytes on the Matrice 400 (22 in dji_fc_subscription.h): two more bytes, apparently health/status flags
    lon, lat, alt, sats, flag0, flag1 = struct.unpack(
        "<ddfHBB", p
    )  # rad, rad, m (WGS84 ellipsoid)
    return {
        "lon_deg": math.degrees(lon),
        "lat_deg": math.degrees(lat),
        "alt_m": alt,
        "fused_nsv": sats,
        "fused_flag0": flag0,
        "fused_flag1": flag1,
    }


def _home_point_info(p):
    lat, lon = struct.unpack("<dd", p)  # rad
    return {"home_lat_deg": math.degrees(lat), "home_lon_deg": math.degrees(lon)}


def _rc_with_flag(p):
    pitch, roll, yaw, thr, flag = struct.unpack("<4fB", p)
    return {
        "rcf_pitch": pitch,
        "rcf_roll": roll,
        "rcf_yaw": yaw,
        "rcf_throttle": thr,
        "rc_logic_connected": bool(flag & 1),
        "rc_sky_connected": bool(flag & 2),
        "rc_ground_connected": bool(flag & 4),
        "rc_app_connected": bool(flag & 8),
    }


def _three_gimbal(p):
    values = struct.unpack("<9f", p)  # 3 x (roll, pitch, yaw), deg
    out = {}
    for i in range(3):
        out.update(
            {
                f"gimbal{i + 1}_roll": values[3 * i],
                f"gimbal{i + 1}_pitch": values[3 * i + 1],
                f"gimbal{i + 1}_yaw": values[3 * i + 2],
            }
        )
    return out


def _imu_atti_navi(p):
    v = struct.unpack("<HH9f4fHHI", p)
    names = (
        "navi_version",
        "navi_flag",
        "navi_pn_x",
        "navi_pn_y",
        "navi_pn_z",
        "navi_vn_x",
        "navi_vn_y",
        "navi_vn_z",
        "navi_an_x",
        "navi_an_y",
        "navi_an_z",
        "navi_q0",
        "navi_q1",
        "navi_q2",
        "navi_q3",
        "navi_resv",
        "navi_cnt",
        "navi_timestamp",
    )
    out = dict(zip(names, v, strict=True))
    del out["navi_resv"]
    return out


def _hard_sync(p):
    v = struct.unpack(
        "<IIIHB4f3f3f", p
    )  # SyncTimestamp, quaternion, accel (g), gyro (rad/s)
    names = (
        "sync_time_2p5ms",
        "sync_time_1ns",
        "sync_reset_time_2p5ms",
        "sync_index",
        "sync_flag",
        "sync_q0",
        "sync_q1",
        "sync_q2",
        "sync_q3",
        "sync_ax",
        "sync_ay",
        "sync_az",
        "sync_gx",
        "sync_gy",
        "sync_gz",
    )
    return dict(zip(names, v, strict=True))


# topic name (from the TOPIC_TABLE record) -> (payload size, decoder).
# Keys shared with the M600 (DJI_M600.utils.decode_broadcast) keep its names and units: the pointing uses
# yaw/pitch/roll and rtk_*; the others (ax/ay/az, gx/gy/gz, mx/my/mz, lat_deg/lon_deg/alt_m, hgt_m, rc_*, rp_*,
# gim_*, landing_gear, ...) keep lager's plots working. Everything else is M400-only.
DECODERS = {
    # imu group
    "quaternion": (16, _quaternion),
    "velocity": (13, _velocity),
    "acceleration_ground": _flat(
        "<3f", ("acc_ground_x", "acc_ground_y", "acc_ground_z")
    ),  # m/s^2, NEU
    "acceleration_body": (12, _acceleration_body),  # g, gravity included
    "acceleration_raw": _flat(
        "<3f", ("acc_raw_x", "acc_raw_y", "acc_raw_z")
    ),  # gravity removed
    "angular_rate_fused": _flat(
        "<3f", ("gyro_fused_x", "gyro_fused_y", "gyro_fused_z")
    ),  # rad/s, NED
    "angular_rate_raw": _flat("<3f", ("gx", "gy", "gz")),  # rad/s, FRD
    "compass": _flat("<3h", ("mx", "my", "mz")),
    "altitude_fused": _flat("<f", ("alt_fused_m",)),
    "altitude_barometer": _flat("<f", ("alt_baro_m",)),
    "height_fusion": _flat("<f", ("hgt_m",)),
    # fast group
    "gps_position": (12, _gps_position),
    "gps_velocity": _flat(
        "<3f", ("gps_vel_x_cms", "gps_vel_y_cms", "gps_vel_z_cms")
    ),  # cm/s
    "rtk_position": (20, _rtk_position),
    "rtk_velocity": _flat(
        "<3f", ("rtk_vel_x_cms", "rtk_vel_y_cms", "rtk_vel_z_cms")
    ),  # cm/s
    "rtk_yaw": (
        4,
        lambda p: {"rtk_yaw_deg": struct.unpack("<f", p)[0]},
    ),  # float deg on the M400
    "rtk_position_info": (1, lambda p: {"rtk_pos_health": p[0]}),
    "rtk_yaw_info": (1, lambda p: {"rtk_yaw_health": p[0]}),
    "position_fused": (24, _position_fused),
    "position_vo": (
        13,
        lambda p: {
            **dict(
                zip(
                    ("vo_x_m", "vo_y_m", "vo_z_m"),
                    struct.unpack_from("<3f", p),
                    strict=True,
                )
            ),
            "vo_health": p[12] & 0x07,
        },
    ),  # NED, m
    "avoid_data": (
        25,
        lambda p: {
            **dict(
                zip(
                    ("rp_down", "rp_front", "rp_right", "rp_back", "rp_left", "rp_up"),
                    struct.unpack_from("<6f", p),
                    strict=True,
                )
            ),
            "avoid_health": p[24] & 0x3F,
        },
    ),  # m
    "rc": _flat(
        "<6h", ("rc_roll", "rc_pitch", "rc_yaw", "rc_thr", "rc_mode", "rc_gear")
    ),
    "rc_with_flag": (17, _rc_with_flag),
    "gimbal_angles": _flat(
        "<3f", ("gim_pitch", "gim_roll", "gim_yaw")
    ),  # deg (x=pitch, y=roll)
    "gimbal_status": _bits(
        "gim_flags",
        4,
        (
            "mounted",
            "busy",
            "pitch_limited",
            "roll_limited",
            "yaw_limited",
            "calibrating",
            "calibration_ok",
            "reversed",
            "",
            "",
            "gyro_fault",
            "esc_pitch_ok",
            "esc_roll_ok",
            "esc_yaw_ok",
            "drone_data_ok",
            "init_complete",
            "fw_updating",
        ),
    ),
    "three_gimbal_data": (36, _three_gimbal),
    # status group
    "gps_details": (36, _gps_details),
    "gps_signal_level": _flat("<B", ("gps_signal_level",)),
    "gps_control_level": _flat("<B", ("gps_control_level",)),
    "flight_status": _flat("<B", ("flight_status",)),
    "display_mode": _flat("<B", ("display_mode",)),
    "landing_gear": _flat("<B", ("landing_gear",)),
    "motor_start_error": _flat("<H", ("motor_start_error",)),
    "control_device": _flat(
        "<BB", ("control_device_0", "control_device_1")
    ),  # meaning depends on the aircraft
    "rtk_connect_status": (2, lambda p: {"rtk_connected": bool(p[0] & 1)}),
    "gimbal_control_mode": _flat("<B", ("gimbal_control_mode",)),
    "flight_anomaly": (4, _flight_anomaly),
    "esc_data": (80, _esc),
    "imu_atti_navi": (64, _imu_atti_navi),
    # slow group
    "battery1": (33, _battery("bat1")),
    "battery2": (33, _battery("bat2")),  # only in older recordings: it repeats battery1
    "gps_date": _flat("<I", ("gps_date",)),  # yyyymmdd
    "gps_time": _flat("<I", ("gps_time",)),  # hhmmss
    "altitude_of_homepoint": _flat("<f", ("home_alt_m",)),
    "height_relative": _flat("<f", ("hgt_relative_m",)),
    "home_point_set_status": _flat("<B", ("home_point_set",)),
    "home_point_info": (16, _home_point_info),
    "hard_sync": (55, _hard_sync),
}


def decode(name: str, payload: bytes, errors: dict | None = None) -> dict:
    """Decode one DATA record.

    Returns {} for an unknown topic, a payload of the wrong size or values that can't be
    decoded (e.g. NaN where an integer is expected): one bad record must never stop the telemetry.

    ``errors`` (optional) counts what was skipped, under ``unknown_topic``, ``bad_size`` and ``bad_value``.
    """

    def skip(kind):
        if errors is not None:
            errors[kind] = errors.get(kind, 0) + 1
        return {}

    entry = DECODERS.get(name)
    if entry is None:
        return skip("unknown_topic")
    if entry[0] != len(payload):
        return skip("bad_size")
    try:
        return entry[1](payload)
    except (ValueError, OverflowError, struct.error):
        return skip("bad_value")
