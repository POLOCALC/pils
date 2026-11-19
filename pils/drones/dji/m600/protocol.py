"""DJI Matrice 600: the OSDK broadcast frames of ``uav_telemetry.bin``.

Ported (copied and adapted, not imported) from LAGER, POLOCALC/lager at
825164a: ``lager/UAVs/DJI_M600/{parameters,utils,DataLoader}.py``. The file
is what LAGER logs from the flight controller's serial link: for each frame
a float64 host time (UNIX seconds), a uint32 length and the DJI OSDK frame.

Changes from LAGER: no live-driver constants; a frame whose CRC32 fails is
dropped and counted (LAGER kept it); no prints.
"""

import math
import struct

CRC_INIT = 0x3AA3
HEADER_LEN = 12
CRC32_LEN = 4
PKG_MIN = HEADER_LEN + CRC32_LEN

# The M600/A3 flight controller runs at 400 Hz: the broadcast's ``t_ms`` is a
# tick counter (1 tick = 2.5 ms), not milliseconds
FC_TICK_MS = 2.5

CMD_SET_BROADCAST = 0x02
CMD_ID_BROADCAST = 0x00

FLAG_STATUS = 0x0800
FLAG_BATTERY = 0x1000
FLAG_DEVICE = 0x2000

# Offsets in the broadcast payload (the bytes after cmd_set and cmd_id)
OFF_FLAG = 0
OFF_TMS = 2  # TimeStamp.time_ms, uint32 (400 Hz ticks)
OFF_TNS = 6  # TimeStamp.time_ns, uint32
OFF_Q = 17  # quaternion q0..q3, 4 x float32
OFF_A = 33  # acceleration, 3 x float32, g
OFF_V = 45  # velocity, 3 x float32, m/s
OFF_VI = 57  # velocity info, uint8
OFF_W = 58  # angular rate, 3 x float32, rad/s
OFF_LAT = 70  # latitude, float64, rad
OFF_LON = 78  # longitude, float64, rad
OFF_ALT = 86  # altitude, float32, m
OFF_HGT = 90  # height above the ground, float32, m
OFF_GPSH = 94  # GPS health, uint8
OFF_RP = 95  # relative position (6 x float32 + uint8 flags)
OFF_GPS_DATE = 120  # GPS date, uint32 yyyymmdd
OFF_GPS_TIME = 124  # GPS time, uint32 hhmmss
OFF_GPS_LON = 128  # GPS longitude, int32, deg x 1e7
OFF_GPS_LAT = 132  # GPS latitude, int32, deg x 1e7
OFF_GPS_HFSL = 136  # GPS height above mean sea level, int32, mm
OFF_GPS_VEL = 140  # GPS velocity NED, 3 x float32, cm/s
OFF_GPS_DTL = 152  # GPS detail
OFF_RTK = 188  # RTK (44 bytes)
OFF_MX = 232  # magnetometer, 3 x int16
OFF_RC = 238  # remote controller, 6 x int16
OFF_GIMBAL = 250  # gimbal, 3 x float32 + uint8
OFF_STATUS = 263  # status, 4 x uint8 (only with FLAG_STATUS)

# CRC32_Common, polynomial 0x04C11DB7 (reflected)
CRC32_TAB = [
    0x00000000,
    0x77073096,
    0xEE0E612C,
    0x990951BA,
    0x076DC419,
    0x706AF48F,
    0xE963A535,
    0x9E6495A3,
    0x0EDB8832,
    0x79DCB8A4,
    0xE0D5E91E,
    0x97D2D988,
    0x09B64C2B,
    0x7EB17CBD,
    0xE7B82D07,
    0x90BF1D91,
    0x1DB71064,
    0x6AB020F2,
    0xF3B97148,
    0x84BE41DE,
    0x1ADAD47D,
    0x6DDDE4EB,
    0xF4D4B551,
    0x83D385C7,
    0x136C9856,
    0x646BA8C0,
    0xFD62F97A,
    0x8A65C9EC,
    0x14015C4F,
    0x63066CD9,
    0xFA0F3D63,
    0x8D080DF5,
    0x3B6E20C8,
    0x4C69105E,
    0xD56041E4,
    0xA2677172,
    0x3C03E4D1,
    0x4B04D447,
    0xD20D85FD,
    0xA50AB56B,
    0x35B5A8FA,
    0x42B2986C,
    0xDBBBC9D6,
    0xACBCF940,
    0x32D86CE3,
    0x45DF5C75,
    0xDCD60DCF,
    0xABD13D59,
    0x26D930AC,
    0x51DE003A,
    0xC8D75180,
    0xBFD06116,
    0x21B4F4B5,
    0x56B3C423,
    0xCFBA9599,
    0xB8BDA50F,
    0x2802B89E,
    0x5F058808,
    0xC60CD9B2,
    0xB10BE924,
    0x2F6F7C87,
    0x58684C11,
    0xC1611DAB,
    0xB6662D3D,
    0x76DC4190,
    0x01DB7106,
    0x98D220BC,
    0xEFD5102A,
    0x71B18589,
    0x06B6B51F,
    0x9FBFE4A5,
    0xE8B8D433,
    0x7807C9A2,
    0x0F00F934,
    0x9609A88E,
    0xE10E9818,
    0x7F6A0DBB,
    0x086D3D2D,
    0x91646C97,
    0xE6635C01,
    0x6B6B51F4,
    0x1C6C6162,
    0x856530D8,
    0xF262004E,
    0x6C0695ED,
    0x1B01A57B,
    0x8208F4C1,
    0xF50FC457,
    0x65B0D9C6,
    0x12B7E950,
    0x8BBEB8EA,
    0xFCB9887C,
    0x62DD1DDF,
    0x15DA2D49,
    0x8CD37CF3,
    0xFBD44C65,
    0x4DB26158,
    0x3AB551CE,
    0xA3BC0074,
    0xD4BB30E2,
    0x4ADFA541,
    0x3DD895D7,
    0xA4D1C46D,
    0xD3D6F4FB,
    0x4369E96A,
    0x346ED9FC,
    0xAD678846,
    0xDA60B8D0,
    0x44042D73,
    0x33031DE5,
    0xAA0A4C5F,
    0xDD0D7CC9,
    0x5005713C,
    0x270241AA,
    0xBE0B1010,
    0xC90C2086,
    0x5768B525,
    0x206F85B3,
    0xB966D409,
    0xCE61E49F,
    0x5EDEF90E,
    0x29D9C998,
    0xB0D09822,
    0xC7D7A8B4,
    0x59B33D17,
    0x2EB40D81,
    0xB7BD5C3B,
    0xC0BA6CAD,
    0xEDB88320,
    0x9ABFB3B6,
    0x03B6E20C,
    0x74B1D29A,
    0xEAD54739,
    0x9DD277AF,
    0x04DB2615,
    0x73DC1683,
    0xE3630B12,
    0x94643B84,
    0x0D6D6A3E,
    0x7A6A5AA8,
    0xE40ECF0B,
    0x9309FF9D,
    0x0A00AE27,
    0x7D079EB1,
    0xF00F9344,
    0x8708A3D2,
    0x1E01F268,
    0x6906C2FE,
    0xF762575D,
    0x806567CB,
    0x196C3671,
    0x6E6B06E7,
    0xFED41B76,
    0x89D32BE0,
    0x10DA7A5A,
    0x67DD4ACC,
    0xF9B9DF6F,
    0x8EBEEFF9,
    0x17B7BE43,
    0x60B08ED5,
    0xD6D6A3E8,
    0xA1D1937E,
    0x38D8C2C4,
    0x4FDFF252,
    0xD1BB67F1,
    0xA6BC5767,
    0x3FB506DD,
    0x48B2364B,
    0xD80D2BDA,
    0xAF0A1B4C,
    0x36034AF6,
    0x41047A60,
    0xDF60EFC3,
    0xA867DF55,
    0x316E8EEF,
    0x4669BE79,
    0xCB61B38C,
    0xBC66831A,
    0x256FD2A0,
    0x5268E236,
    0xCC0C7795,
    0xBB0B4703,
    0x220216B9,
    0x5505262F,
    0xC5BA3BBE,
    0xB2BD0B28,
    0x2BB45A92,
    0x5CB36A04,
    0xC2D7FFA7,
    0xB5D0CF31,
    0x2CD99E8B,
    0x5BDEAE1D,
    0x9B64C2B0,
    0xEC63F226,
    0x756AA39C,
    0x026D930A,
    0x9C0906A9,
    0xEB0E363F,
    0x72076785,
    0x05005713,
    0x95BF4A82,
    0xE2B87A14,
    0x7BB12BAE,
    0x0CB61B38,
    0x92D28E9B,
    0xE5D5BE0D,
    0x7CDCEFB7,
    0x0BDBDF21,
    0x86D3D2D4,
    0xF1D4E242,
    0x68DDB3F8,
    0x1FDA836E,
    0x81BE16CD,
    0xF6B9265B,
    0x6FB077E1,
    0x18B74777,
    0x88085AE6,
    0xFF0F6A70,
    0x66063BCA,
    0x11010B5C,
    0x8F659EFF,
    0xF862AE69,
    0x616BFFD3,
    0x166CCF45,
    0xA00AE278,
    0xD70DD2EE,
    0x4E048354,
    0x3903B3C2,
    0xA7672661,
    0xD06016F7,
    0x4969474D,
    0x3E6E77DB,
    0xAED16A4A,
    0xD9D65ADC,
    0x40DF0B66,
    0x37D83BF0,
    0xA9BCAE53,
    0xDEBB9EC5,
    0x47B2CF7F,
    0x30B5FFE9,
    0xBDBDF21C,
    0xCABAC28A,
    0x53B39330,
    0x24B4A3A6,
    0xBAD03605,
    0xCDD70693,
    0x54DE5729,
    0x23D967BF,
    0xB3667A2E,
    0xC4614AB8,
    0x5D681B02,
    0x2A6F2B94,
    0xB40BBE37,
    0xC30C8EA1,
    0x5A05DF1B,
    0x2D02EF8D,
]


def crc32(data: bytes) -> int:
    """DJI's CRC-32 (the reflected table above, initial value ``CRC_INIT``, no final xor)."""
    crc = CRC_INIT
    for b in data:
        crc = (crc >> 8) ^ CRC32_TAB[(crc ^ b) & 0xFF]
    return crc & 0xFFFFFFFF


def quat_to_euler(
    q0: float, q1: float, q2: float, q3: float
) -> tuple[float, float, float]:
    """Quaternion (q0 = w, q1 = x, q2 = y, q3 = z) to (roll, pitch, yaw) in degrees."""
    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    sinp = 2 * (q0 * q2 - q3 * q1)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def parse_frame(raw: bytes) -> dict:
    """Split a frame into its header fields and payload.

    Returns ``length``, ``is_ack``, ``seq``, ``cmd_set`` and ``cmd_id`` (None
    for an ACK), ``payload`` and ``crc32_ok``. ``raw`` must hold at least the
    header and the CRC32.
    """
    (w0,) = struct.unpack_from("<I", raw, 0)
    length = (w0 >> 8) & 0x3FF
    is_ack = bool((w0 >> 29) & 1)
    (seq,) = struct.unpack_from("<H", raw, 8)
    if length < PKG_MIN or length > len(raw):
        return {
            "length": length,
            "is_ack": is_ack,
            "seq": seq,
            "cmd_set": None,
            "cmd_id": None,
            "payload": b"",
            "crc32_ok": False,
        }
    crc32_ok = (
        crc32(raw[: length - CRC32_LEN])
        == struct.unpack_from("<I", raw, length - CRC32_LEN)[0]
    )
    if is_ack:
        cmd_set, cmd_id = None, None
        payload = raw[HEADER_LEN : length - CRC32_LEN]
    else:
        cmd_set = raw[HEADER_LEN]
        cmd_id = raw[HEADER_LEN + 1]
        payload = raw[HEADER_LEN + 2 : length - CRC32_LEN]
    return {
        "length": length,
        "is_ack": is_ack,
        "seq": seq,
        "cmd_set": cmd_set,
        "cmd_id": cmd_id,
        "payload": payload,
        "crc32_ok": crc32_ok,
    }


def decode_broadcast(data: bytes) -> dict | None:
    """Decode a broadcast payload (cmd_set 0x02, cmd_id 0x00).

    Two kinds of frame: the full telemetry (flag 0x07FF or 0x0FFF, 263 or
    267 bytes) and the battery + SDK info only (flag 0x3000, 17 bytes). None
    if the payload is too short.
    """
    if len(data) < 4:
        return None
    (flag,) = struct.unpack_from("<H", data, OFF_FLAG)

    if flag == (FLAG_BATTERY | FLAG_DEVICE):
        off_bat, off_sdk = 2, 15
        if len(data) < off_sdk + 2:
            return None
        bat_cap, bat_v, bat_i, bat_pct = struct.unpack_from("<IiiB", data, off_bat)
        sdk_ctrl, sdk_dev = struct.unpack_from("<BB", data, off_sdk)
        return {
            "flag": flag,
            "bat_capacity_mah": bat_cap,
            "bat_voltage_mv": bat_v,
            "bat_current_ma": bat_i,
            "bat_percentage": bat_pct,
            "sdk_control_mode": sdk_ctrl,
            "sdk_device_bits": sdk_dev,
        }

    if len(data) < OFF_STATUS:
        return None

    (t_ticks,) = struct.unpack_from("<I", data, OFF_TMS)
    (t_ns,) = struct.unpack_from("<I", data, OFF_TNS)
    q0, q1, q2, q3 = struct.unpack_from("<4f", data, OFF_Q)
    roll, pitch, yaw = quat_to_euler(q0, q1, q2, q3)
    ax, ay, az = struct.unpack_from("<3f", data, OFF_A)
    vx, vy, vz = struct.unpack_from("<3f", data, OFF_V)
    gx, gy, gz = struct.unpack_from("<3f", data, OFF_W)
    (lat_rad,) = struct.unpack_from("<d", data, OFF_LAT)
    (lon_rad,) = struct.unpack_from("<d", data, OFF_LON)
    rp = struct.unpack_from("<6f", data, OFF_RP)
    gps_date, gps_time = struct.unpack_from("<II", data, OFF_GPS_DATE)
    gps_lon, gps_lat, gps_hfsl_mm = struct.unpack_from("<3i", data, OFF_GPS_LON)
    gps_vn, gps_ve, gps_vd = struct.unpack_from("<3f", data, OFF_GPS_VEL)
    hdop, pdop = struct.unpack_from("<2f", data, OFF_GPS_DTL)
    (gps_nsv,) = struct.unpack_from("<H", data, OFF_GPS_DTL + 32)
    rtk_lon, rtk_lat = struct.unpack_from("<2d", data, OFF_RTK + 8)
    (rtk_hfsl,) = struct.unpack_from("<f", data, OFF_RTK + 24)
    (rtk_yaw,) = struct.unpack_from("<h", data, OFF_RTK + 40)
    mx, my, mz = struct.unpack_from("<3h", data, OFF_MX)
    rc = struct.unpack_from("<6h", data, OFF_RC)
    gim_roll, gim_pitch, gim_yaw = struct.unpack_from("<3f", data, OFF_GIMBAL)

    out = {
        "flag": flag,
        "t_ticks": t_ticks,
        "t_ns": t_ns,
        "q0": q0,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "qmag": math.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "ax": ax,
        "ay": ay,
        "az": az,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "vel_info": data[OFF_VI],
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "lat_deg": math.degrees(lat_rad),
        "lon_deg": math.degrees(lon_rad),
        "alt_m": struct.unpack_from("<f", data, OFF_ALT)[0],
        "hgt_m": struct.unpack_from("<f", data, OFF_HGT)[0],
        "gps_health": data[OFF_GPSH],
        "rp_down": rp[0],
        "rp_front": rp[1],
        "rp_right": rp[2],
        "rp_back": rp[3],
        "rp_left": rp[4],
        "rp_up": rp[5],
        "gps_date": gps_date,
        "gps_time": gps_time,
        "gps_lon_deg": gps_lon / 1e7,
        "gps_lat_deg": gps_lat / 1e7,
        "gps_hfsl_m": gps_hfsl_mm / 1000.0,
        "gps_vn_cms": gps_vn,
        "gps_ve_cms": gps_ve,
        "gps_vd_cms": gps_vd,
        "hdop": hdop,
        "pdop": pdop,
        "gps_nsv": gps_nsv,
        "rtk_lon_deg": rtk_lon,
        "rtk_lat_deg": rtk_lat,
        "rtk_hfsl_m": rtk_hfsl,
        "rtk_yaw_deg": rtk_yaw * 0.1,
        "rtk_pos_health": data[OFF_RTK + 42],
        "rtk_yaw_health": data[OFF_RTK + 43],
        "mx": mx,
        "my": my,
        "mz": mz,
        "rc_roll": rc[0],
        "rc_pitch": rc[1],
        "rc_yaw": rc[2],
        "rc_thr": rc[3],
        "rc_mode": rc[4],
        "rc_gear": rc[5],
        "gim_roll": gim_roll,
        "gim_pitch": gim_pitch,
        "gim_yaw": gim_yaw,
        "gim_flags": data[OFF_GIMBAL + 12],
    }
    if (flag & FLAG_STATUS) and len(data) >= OFF_STATUS + 4:
        status, mode, gear, error = struct.unpack_from("<4B", data, OFF_STATUS)
        out.update(
            {
                "flight_status": status,
                "display_mode": mode,
                "landing_gear": gear,
                "flight_error": error,
            }
        )
    return out
