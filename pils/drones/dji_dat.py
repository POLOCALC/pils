"""The binary DJI .dat flight records: record layouts, CRC and vectorized decoding."""

from typing import Any

import numpy as np
import polars as pl

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


CRC_TABLE = [
    0x0000,
    0x1189,
    0x2312,
    0x329B,
    0x4624,
    0x57AD,
    0x6536,
    0x74BF,
    0x8C48,
    0x9DC1,
    0xAF5A,
    0xBED3,
    0xCA6C,
    0xDBE5,
    0xE97E,
    0xF8F7,
    0x1081,
    0x0108,
    0x3393,
    0x221A,
    0x56A5,
    0x472C,
    0x75B7,
    0x643E,
    0x9CC9,
    0x8D40,
    0xBFDB,
    0xAE52,
    0xDAED,
    0xCB64,
    0xF9FF,
    0xE876,
    0x2102,
    0x308B,
    0x0210,
    0x1399,
    0x6726,
    0x76AF,
    0x4434,
    0x55BD,
    0xAD4A,
    0xBCC3,
    0x8E58,
    0x9FD1,
    0xEB6E,
    0xFAE7,
    0xC87C,
    0xD9F5,
    0x3183,
    0x200A,
    0x1291,
    0x0318,
    0x77A7,
    0x662E,
    0x54B5,
    0x453C,
    0xBDCB,
    0xAC42,
    0x9ED9,
    0x8F50,
    0xFBEF,
    0xEA66,
    0xD8FD,
    0xC974,
    0x4204,
    0x538D,
    0x6116,
    0x709F,
    0x0420,
    0x15A9,
    0x2732,
    0x36BB,
    0xCE4C,
    0xDFC5,
    0xED5E,
    0xFCD7,
    0x8868,
    0x99E1,
    0xAB7A,
    0xBAF3,
    0x5285,
    0x430C,
    0x7197,
    0x601E,
    0x14A1,
    0x0528,
    0x37B3,
    0x263A,
    0xDECD,
    0xCF44,
    0xFDDF,
    0xEC56,
    0x98E9,
    0x8960,
    0xBBFB,
    0xAA72,
    0x6306,
    0x728F,
    0x4014,
    0x519D,
    0x2522,
    0x34AB,
    0x0630,
    0x17B9,
    0xEF4E,
    0xFEC7,
    0xCC5C,
    0xDDD5,
    0xA96A,
    0xB8E3,
    0x8A78,
    0x9BF1,
    0x7387,
    0x620E,
    0x5095,
    0x411C,
    0x35A3,
    0x242A,
    0x16B1,
    0x0738,
    0xFFCF,
    0xEE46,
    0xDCDD,
    0xCD54,
    0xB9EB,
    0xA862,
    0x9AF9,
    0x8B70,
    0x8408,
    0x9581,
    0xA71A,
    0xB693,
    0xC22C,
    0xD3A5,
    0xE13E,
    0xF0B7,
    0x0840,
    0x19C9,
    0x2B52,
    0x3ADB,
    0x4E64,
    0x5FED,
    0x6D76,
    0x7CFF,
    0x9489,
    0x8500,
    0xB79B,
    0xA612,
    0xD2AD,
    0xC324,
    0xF1BF,
    0xE036,
    0x18C1,
    0x0948,
    0x3BD3,
    0x2A5A,
    0x5EE5,
    0x4F6C,
    0x7DF7,
    0x6C7E,
    0xA50A,
    0xB483,
    0x8618,
    0x9791,
    0xE32E,
    0xF2A7,
    0xC03C,
    0xD1B5,
    0x2942,
    0x38CB,
    0x0A50,
    0x1BD9,
    0x6F66,
    0x7EEF,
    0x4C74,
    0x5DFD,
    0xB58B,
    0xA402,
    0x9699,
    0x8710,
    0xF3AF,
    0xE226,
    0xD0BD,
    0xC134,
    0x39C3,
    0x284A,
    0x1AD1,
    0x0B58,
    0x7FE7,
    0x6E6E,
    0x5CF5,
    0x4D7C,
    0xC60C,
    0xD785,
    0xE51E,
    0xF497,
    0x8028,
    0x91A1,
    0xA33A,
    0xB2B3,
    0x4A44,
    0x5BCD,
    0x6956,
    0x78DF,
    0x0C60,
    0x1DE9,
    0x2F72,
    0x3EFB,
    0xD68D,
    0xC704,
    0xF59F,
    0xE416,
    0x90A9,
    0x8120,
    0xB3BB,
    0xA232,
    0x5AC5,
    0x4B4C,
    0x79D7,
    0x685E,
    0x1CE1,
    0x0D68,
    0x3FF3,
    0x2E7A,
    0xE70E,
    0xF687,
    0xC41C,
    0xD595,
    0xA12A,
    0xB0A3,
    0x8238,
    0x93B1,
    0x6B46,
    0x7ACF,
    0x4854,
    0x59DD,
    0x2D62,
    0x3CEB,
    0x0E70,
    0x1FF9,
    0xF78F,
    0xE606,
    0xD49D,
    0xC514,
    0xB1AB,
    0xA022,
    0x92B9,
    0x8330,
    0x7BC7,
    0x6A4E,
    0x58D5,
    0x495C,
    0x3DE3,
    0x2C6A,
    0x1EF1,
    0x0F78,
]


def calc_checksum(memory: bytes, start: int, plength: int) -> int:
    """Compute the CRC16 of a DJI .dat record (seed 0x3692)."""
    v = 0x3692  # P3

    for i in range(plength):
        byte_val = memory[start + i]
        v = CRC_TABLE[(byte_val ^ (v & 0xFF)) & 0xFF] ^ (v >> 8)
        v &= 0xFFFF  # keep 16-bit behavior

    return v


# Message type definitions with their struct formats and field mappings
MESSAGE_DEFINITIONS = {
    2096: {  # GPS data
        "name": "GPS",
        "payload_size": 66,
        "fields": [
            ("date", "I", 0),
            ("time", "I", 4),
            ("longitude", "i", 8, lambda x: x / 1.0e7),
            ("latitude", "i", 12, lambda x: x / 1.0e7),
            ("heightMSL", "i", 16, lambda x: x / 1000.0),
            ("velN", "f", 20, lambda x: x / 100.0),
            ("velE", "f", 24, lambda x: x / 100.0),
            ("velD", "f", 28, lambda x: x / 100.0),
            # DOPs and accuracies are stored x100 (checked against DatCon 4.3.0,
            # except hacc, which DatCon does not export)
            ("hdop", "f", 32, lambda x: x / 100.0),
            ("pdop", "f", 36, lambda x: x / 100.0),
            ("hacc", "f", 40, lambda x: x / 100.0),
            ("sacc", "f", 44, lambda x: x / 100.0),
            ("numGPS", "I", 56),
            ("numGLN", "I", 60),
            ("numSV", "H", 64),
        ],
    },
    53234: {  # RTK data; layout checked against DatCon 4.3.0's RTKdata columns
        "name": "RTK",
        "payload_size": 72,
        "fields": [
            ("date", "I", 0),
            ("time", "I", 4),
            ("lon_p", "d", 8),
            ("lat_p", "d", 16),
            ("hmsl_p", "f", 24),
            ("lon_s", "i", 28, lambda x: x / 1.0e7),
            ("lat_s", "i", 32, lambda x: x / 1.0e7),
            ("hmsl_s", "i", 36, lambda x: x / 1000.0),
            ("vel_n", "f", 40, lambda x: x / 100.0),
            ("vel_e", "f", 44, lambda x: x / 100.0),
            ("vel_d", "f", 48, lambda x: x / 100.0),
            ("yaw", "h", 52),
            ("svn_s", "B", 54),
            ("svn_p", "B", 55),
            ("hdop", "f", 56, lambda x: x / 100.0),
            ("pitch", "f", 60),
            ("pos_flg_0", "B", 64),
            ("pos_flg_1", "B", 65),
            ("pos_flg_2", "B", 66),
            ("pos_flg_3", "B", 67),
            ("pos_flg_4", "B", 68),
            ("pos_flg_5", "B", 69),
            ("gps_state", "H", 70),
        ],
    },
}


# Record layouts beyond GPS/RTK, from DatCon's V3 record classes
# (https://github.com/BudWalkerJava/DatCon, src/DatConRecs) and checked against
# DatCon's CSV of the same flight, tick by tick. Names follow DatCon's columns.
# Fields: (name, struct format, byte offset in the payload, optional conversion)
_IMU_FIELDS = [
    ("longitude", "d", 0, np.degrees),
    ("latitude", "d", 8, np.degrees),
    ("press", "f", 16),  # barometric altitude (m)
    ("accelX", "f", 20),  # g
    ("accelY", "f", 24),
    ("accelZ", "f", 28),
    ("gyroX", "f", 32, np.degrees),  # deg/s
    ("gyroY", "f", 36, np.degrees),
    ("gyroZ", "f", 40, np.degrees),
    ("alti", "f", 44),  # fused altitude (m)
    ("quatW", "f", 48),
    ("quatX", "f", 52),
    ("quatY", "f", 56),
    ("quatZ", "f", 60),
    ("agX", "f", 64),
    ("agY", "f", 68),
    ("agZ", "f", 72),
    ("velN", "f", 76),  # m/s
    ("velE", "f", 80),
    ("velD", "f", 84),
    ("gbX", "f", 88),  # gyro biases
    ("gbY", "f", 92),
    ("gbZ", "f", 96),
    ("magX", "h", 100),
    ("magY", "h", 102),
    ("magZ", "h", 104),
    ("temperature", "h", 106, lambda x: x / 100.0),  # deg C
    ("numSats", "B", 116),
]
_GPS_FIELDS = MESSAGE_DEFINITIONS[2096]["fields"]
_MOTORS = ["RFront", "LFront", "LSide", "LBack", "RBack", "RSide"]  # hexacopter

DAT_RECORDS: dict[int, dict[str, Any]] = {
    2048: {"name": "IMU_ATTI(0)", "fields": _IMU_FIELDS, "euler": True},
    2049: {"name": "IMU_ATTI(1)", "fields": _IMU_FIELDS, "euler": True},
    2050: {"name": "IMU_ATTI(2)", "fields": _IMU_FIELDS, "euler": True},
    2097: {"name": "GPS(1)", "fields": _GPS_FIELDS},
    2098: {"name": "GPS(2)", "fields": _GPS_FIELDS},
    12: {
        "name": "osd_data",
        "fields": [
            ("relativeHeight", "h", 16, lambda x: x / 10.0),  # m
            ("vpsHeight", "B", 41, lambda x: x / 10.0),  # m
            ("flightTime", "h", 42, lambda x: x * 100.0),  # ms
        ],
    },
    1000: {
        "name": "Controller",
        "fields": [
            ("ctrl_tick", "I", 0),
            ("ctrl_pitch", "h", 4),
            ("ctrl_roll", "h", 6),
            ("ctrl_yaw", "h", 8),
            ("ctrl_thr", "h", 10),
            ("mode_switch", "B", 13),
            ("motor_state", "B", 14),
            ("sig_level", "B", 15),
            ("max_height", "H", 18),
            ("max_radius", "H", 20),
            ("vert_vel_err", "f", 44),
        ],
    },
    1710: {  # physical units; DatCon's CSV has the raw counts (mV, mA, 0.1 C)
        "name": "BatteryInfo",
        "fields": [
            ("ad_v", "H", 0, lambda x: x / 1000.0),  # V
            ("r_time", "H", 2),  # s
            ("ave_I", "f", 4, lambda x: x / 1000.0),  # A
            ("vol_t", "f", 8),
            ("pack_ve", "i", 12, lambda x: x / 1000.0),  # V
            ("I", "i", 16, lambda x: -x / 1000.0),  # A
            ("r_cap", "H", 20),
            ("cap_per", "B", 22),  # %
            ("temp", "h", 23, lambda x: x / 10.0),  # deg C
            ("l_cell", "H", 26),
            ("f_cap", "I", 32),
        ],
    },
    10100: {
        "name": "AirSpeed",
        "fields": [
            ("air_vbx", "f", 0),
            ("air_vby", "f", 4),
            ("comp_alti", "f", 8),
            ("windSpeed", "f", 12),  # m/s
            ("windX", "f", 16),  # not in DatCon's CSV (it gives windN/windE)
            ("windY", "f", 20),
            ("MotorSpd", "f", 24),
        ],
    },
    1307: {
        "name": "MotorCtrl",
        "fields": [
            (f"PWM:{pos}", "H", 2 * i, lambda x: x / 100.0)  # %
            for i, pos in enumerate(_MOTORS)
        ],
    },
    10090: {
        "name": "Motor",
        "fields": [
            field
            for i, pos in enumerate(_MOTORS)
            for field in (
                (f"Status:{pos}", "B", 23 * i),
                (f"Current:{pos}", "h", 23 * i + 1, lambda x: x / 100.0),  # A
                (f"Speed:{pos}", "h", 23 * i + 3),  # rpm
                (f"Volts:{pos}", "h", 23 * i + 5, lambda x: x / 10.0),
                (f"EscTemp:{pos}", "h", 23 * i + 7),  # deg C
                (f"PPMrecv:{pos}", "h", 23 * i + 9, lambda x: x / 19.2),  # %
                (f"V_out:{pos}", "h", 23 * i + 11, lambda x: x / 10.0),
                (f"PPMsend:{pos}", "H", 23 * i + 19, lambda x: x / 19.2),  # %
            )
        ],
    },
}

_NUMPY_FORMATS = {
    "d": "<f8",
    "f": "<f4",
    "i": "<i4",
    "I": "<u4",
    "h": "<i2",
    "H": "<u2",
    "b": "i1",
    "B": "u1",
}
_CRC_TABLE_NP = np.array(CRC_TABLE, dtype=np.uint16)
_HEADER_SIZE = 10  # 0x55, length, 2 bytes, type (uint16), tick (uint32)


def scan_records(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the records of a DJI V3 .dat file.

    A record is ``0x55``, its length, two bytes, the message type (uint16),
    the tick (uint32), the payload XOR-ed with the tick's low byte, and a
    CRC-16 of everything before it. Every ``0x55`` byte is a candidate start;
    candidates whose checksum matches are kept, and ones that fall inside an
    accepted record are dropped (the byte also occurs inside payloads).

    Parameters
    ----------
    data : bytes
        Contents of the .dat file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The file as uint8, and the start and length of each record.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    candidates = np.flatnonzero(arr == 0x55)
    candidates = candidates[candidates + _HEADER_SIZE + 2 <= arr.size]
    lengths = arr[candidates + 1].astype(np.int64)
    fits = (lengths >= _HEADER_SIZE + 2) & (candidates + lengths <= arr.size)
    candidates, lengths = candidates[fits], lengths[fits]

    valid = np.zeros(candidates.size, dtype=bool)
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        # Bounded memory: the body of up to 200k records at a time
        for chunk in np.array_split(rows, rows.size // 200_000 + 1):
            starts = candidates[chunk]
            body = arr[starts[:, None] + np.arange(length - 2)]
            crc = np.full(starts.size, 0x3692, dtype=np.uint16)
            for column in body.T:
                crc = _CRC_TABLE_NP[(column ^ crc) & 0xFF] ^ (crc >> 8)
            stored = arr[starts + length - 2].astype(np.uint16) | (
                arr[starts + length - 1].astype(np.uint16) << 8
            )
            valid[chunk] = crc == stored
    candidates, lengths = candidates[valid], lengths[valid]

    keep = np.zeros(candidates.size, dtype=bool)
    next_free = 0
    for i, (start, length) in enumerate(
        zip(candidates.tolist(), lengths.tolist(), strict=True)
    ):
        if start >= next_free:
            keep[i] = True
            next_free = start + length
    return arr, candidates[keep], lengths[keep]


def unwrap_ticks(ticks: np.ndarray, wrap_threshold: float = 1e8) -> np.ndarray:
    """Unwrap the uint32 ticks of all records, in file order.

    All record types share the flight controller clock, but records of
    different types are written slightly out of tick order, so only large
    jumps count: a drop of more than 2^31 is a wrap (+2^32) and a rise of
    more than 2^31 is a record from before the wrap written after it (-2^32).
    A drop of more than ``wrap_threshold`` to below ``wrap_threshold`` that is
    not a wrap is a clock reset, as in :meth:`DJIDrone._unwrap_tick`.
    """
    ticks = ticks.astype(np.int64)
    if ticks.size < 2:
        return ticks
    diff = np.diff(ticks)
    steps = np.zeros(ticks.size, dtype=np.int64)
    steps[1:] += np.where(diff < -(2**31), 2**32, 0)
    steps[1:] -= np.where(diff > 2**31, 2**32, 0)
    resets = np.flatnonzero(
        (diff < -wrap_threshold) & (diff >= -(2**31)) & (ticks[1:] < wrap_threshold)
    )
    steps[resets + 1] += ticks[resets]
    return ticks + np.cumsum(steps)


def decode_records(
    arr: np.ndarray,
    starts: np.ndarray,
    lengths: np.ndarray,
    ticks: np.ndarray,
    name: str,
    fields: list[tuple],
) -> pl.DataFrame:
    """Decode records of one type (all at once) into ``name:field`` columns.

    Parameters
    ----------
    arr : np.ndarray
        The file as uint8.
    starts, lengths, ticks : np.ndarray
        Start, length and (unwrapped) tick of the records of this type.
    name : str
        Record name, the column prefix.
    fields : list of tuple
        ``(field, struct format, offset[, conversion])``.
    """
    size = max(f[2] + np.dtype(_NUMPY_FORMATS[f[1]]).itemsize for f in fields)
    ok = lengths - _HEADER_SIZE - 2 >= size
    starts, ticks = starts[ok], ticks[ok]
    key = arr[starts + 6]  # low byte of the tick
    payload = arr[starts[:, None] + _HEADER_SIZE + np.arange(size)] ^ key[:, None]
    view = np.ascontiguousarray(payload).view(
        np.dtype(
            {
                "names": [f[0] for f in fields],
                "formats": [_NUMPY_FORMATS[f[1]] for f in fields],
                "offsets": [f[2] for f in fields],
                "itemsize": size,
            }
        )
    )[:, 0]
    columns: dict[str, Any] = {}
    # Corrupt or unused float fields can hold signalling-NaN bit patterns
    with np.errstate(invalid="ignore"):
        for field in fields:
            values = view[field[0]]
            if len(field) > 3 and callable(field[3]):
                values = field[3](values.astype(np.float64))
            elif values.dtype.kind in "iu":
                values = values.astype(np.int64)
            else:
                values = values.astype(np.float64)
            columns[f"{name}:{field[0]}"] = values
    columns["tick"] = ticks
    return pl.DataFrame(columns)


def add_euler_angles(df: pl.DataFrame, name: str) -> pl.DataFrame:
    """Roll, pitch and yaw (degrees) from the IMU quaternion, as DatCon.

    DatCon's convention: ``roll = asin(2(wx + yz))`` and
    ``pitch = atan2(2(wy - xz), 1 - 2(x^2 + y^2))``, with the gimbal-lock
    branches of its ``Quaternion.toEuler``.
    """
    w, x, y, z = (df[f"{name}:quat{c}"].to_numpy() for c in "WXYZ")
    unit = w * w + x * x + y * y + z * z
    test = w * x + y * z
    with np.errstate(invalid="ignore", divide="ignore"):
        yaw = np.arctan2(2 * (w * z - x * y), 1 - 2 * (z * z + x * x))
        roll = np.arcsin(np.clip(2 * test / unit, -1, 1))
        pitch = np.arctan2(2 * (w * y - x * z), 1 - 2 * (y * y + x * x))
    north, south = test > 0.499 * unit, test < -0.499 * unit
    yaw = np.where(
        north, 2 * np.arctan2(y, w), np.where(south, -2 * np.arctan2(y, w), yaw)
    )
    pitch = np.where(north, np.pi / 2, np.where(south, -np.pi / 2, pitch))
    roll = np.where(north | south, 0.0, roll)
    return df.with_columns(
        pl.Series(f"{name}:roll", np.degrees(roll)),
        pl.Series(f"{name}:pitch", np.degrees(pitch)),
        pl.Series(f"{name}:yaw", np.degrees(yaw)),
    )
