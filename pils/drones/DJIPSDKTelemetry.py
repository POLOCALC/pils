"""
Reader for ``uav_telemetry.bin`` written by the payload computer through the
DJI Payload SDK (Matrice 400). Used as a fallback when the aircraft's own
``.DAT`` flight log is missing or can't be decrypted.

Stream format and payload layouts follow lager (``lager/UAVs/DJI_M400`` on
the ``porter_integration`` branch: ``stream.py``, ``parameters.py``,
``utils.py``), trimmed to the topics needed here.

Record layout (little endian)::

    A5 5A | version u8 | type u8 | topic u16 | length u16 | host_ns i64
          | aircraft_ms u32 | aircraft_us u32 | payload[length] | crc16 u16

``host_ns`` is the payload computer's clock (UTC), the same clock that
timestamps the Alvium frames.
"""

import binascii
import logging
import math
import struct
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

SYNC = b"\xa5\x5a"
HEADER = struct.Struct("<BBHHqII")  # after SYNC
HEADER_SIZE = 2 + HEADER.size  # 24
CRC_SIZE = 2
MAX_PAYLOAD = 2048
STREAM_VERSION = 1
TOPIC_NAME_SIZE = 32
TOPIC_ENTRY_SIZE = 7 + TOPIC_NAME_SIZE  # u16 topic, u16 Hz, u16 size, u8 subscribed, name[32]

RECORD_DATA, RECORD_STATUS, RECORD_TOPIC_TABLE = 1, 2, 3

FLIGHT_STATUS_IN_AIR = 2  # PSDK: 0 stopped, 1 on ground, 2 in air


def _quaternion(p):
    q0, q1, q2, q3 = struct.unpack("<4f", p)
    roll = math.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
    sinp = 2 * (q0 * q2 - q3 * q1)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    return {
        "q0": q0, "q1": q1, "q2": q2, "q3": q3,
        "roll": math.degrees(roll), "pitch": math.degrees(pitch), "yaw": math.degrees(yaw),
    }


def _velocity(p):
    vx, vy, vz, _ = struct.unpack("<3fB", p)  # m/s, North-East-Up
    return {"vel_n": vx, "vel_e": vy, "vel_u": vz}


def _gps_position(p):
    lon, lat, alt = struct.unpack("<3i", p)  # deg*1e7, deg*1e7, mm
    return {"GPS:longitude": lon * 1e-7, "GPS:latitude": lat * 1e-7, "GPS:heightMSL": alt / 1000.0}


def _gps_details(p):
    hdop, pdop, fix, _, hacc, _, n_gps, n_glonass, n_total, _ = struct.unpack("<6f2I2H", p)
    return {
        "GPS:hdop": hdop * 0.01, "GPS:pdop": pdop * 0.01,  # PSDK: 100 = 1.00
        "GPS:fix": int(fix) if math.isfinite(fix) else None, "GPS:hacc": hacc,
        "GPS:numGPS": n_gps, "GPS:numGLN": n_glonass, "GPS:numSV": n_total,
    }


def _rtk_position(p):
    lon, lat, hfsl = struct.unpack("<ddf", p)  # deg, deg, m above sea level
    return {"RTK:lon_p": lon, "RTK:lat_p": lat, "RTK:hmsl_p": hfsl}


def _position_fused(p):
    # 24 bytes on the Matrice 400 (22 in dji_fc_subscription.h)
    lon, lat, alt, sats, _, _ = struct.unpack("<ddfHBB", p)  # rad, rad, m (WGS84 ellipsoid)
    return {"fused:lon": math.degrees(lon), "fused:lat": math.degrees(lat),
            "fused:alt": alt, "fused:numSV": sats}


def _single(fmt, name):
    return lambda p: {name: struct.unpack(fmt, p)[0]}


# topic name -> (payload size, decoder)
DECODERS = {
    "quaternion":        (16, _quaternion),
    "velocity":          (13, _velocity),
    "gps_position":      (12, _gps_position),
    "gps_details":       (36, _gps_details),
    "rtk_position":      (20, _rtk_position),
    "rtk_yaw":           (4, _single("<f", "RTK:yaw")),
    "rtk_position_info": (1, _single("<B", "RTK:pos_health")),
    "position_fused":    (24, _position_fused),
    "altitude_fused":    (4, _single("<f", "alt_fused")),
    "height_fusion":     (4, _single("<f", "height_fusion")),
    "flight_status":     (1, _single("<B", "flight_status")),
}


def _crc16(data) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF), as written by the logger."""
    return binascii.crc_hqx(data, 0xFFFF)


def is_psdk_telemetry(path: str | Path) -> bool:
    """True if ``path`` starts with a PSDK stream record (the M600 log starts with a float64 time)."""
    try:
        with open(path, "rb") as f:
            head = f.read(HEADER_SIZE)
    except OSError:
        return False
    return len(head) == HEADER_SIZE and head[:2] == SYNC and head[2] == STREAM_VERSION


def read_records(data: bytes):
    """Yield (type, topic, host_ns, payload) for every record with a valid CRC."""
    pos, n = 0, len(data)
    bad_crc = 0
    while True:
        start = data.find(SYNC, pos)
        if start < 0 or n - start < HEADER_SIZE:
            break
        version, rtype, topic, length, host_ns, _, _ = HEADER.unpack_from(data, start + 2)
        if version != STREAM_VERSION or length > MAX_PAYLOAD:
            pos = start + 1
            continue
        end = start + HEADER_SIZE + length + CRC_SIZE
        if end > n:
            break
        (crc,) = struct.unpack_from("<H", data, end - CRC_SIZE)
        if _crc16(data[start + 2:end - CRC_SIZE]) != crc:
            bad_crc += 1
            pos = start + 1
            continue
        yield rtype, topic, host_ns, data[start + HEADER_SIZE:end - CRC_SIZE]
        pos = end
    if bad_crc:
        logger.warning(f"{bad_crc} telemetry records skipped (bad CRC)")


def _parse_topic_table(payload: bytes) -> dict[int, str]:
    (count,) = struct.unpack_from("<H", payload, 0)
    count = min(count, (len(payload) - 2) // TOPIC_ENTRY_SIZE)
    topics = {}
    for i in range(count):
        off = 2 + i * TOPIC_ENTRY_SIZE
        (tid,) = struct.unpack_from("<H", payload, off)
        topics[tid] = payload[off + 7:off + TOPIC_ENTRY_SIZE].split(b"\0")[0].decode()
    return topics


class DJIPSDKTelemetry:
    """
    Drone telemetry from ``uav_telemetry.bin`` (DJI Payload SDK, Matrice 400).

    ``load_data()`` sets ``self.data`` to a polars DataFrame with one row per
    position fix (RTK when available, otherwise GPS), every other topic
    joined as its latest value at that time. Column names match
    ``DJIDrone``'s DAT output where the quantity is the same (``timestamp``,
    ``datetime``, ``RTK:lat_p``, ``RTK:lon_p``, ``RTK:hmsl_p``, ``GPS:*``),
    plus ``roll``/``pitch``/``yaw`` (deg), ``q0..q3``, ``vel_n/e/u`` (m/s),
    ``flight_status`` and ``isflying`` (as in the Litchi log).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: pl.DataFrame | None = None

    def load_data(self) -> None:
        data = self.path.read_bytes()

        topics: dict[int, str] = {}
        rows: dict[str, list[dict]] = {}
        for rtype, topic, host_ns, payload in read_records(data):
            if rtype == RECORD_TOPIC_TABLE:
                topics = _parse_topic_table(payload)  # rewritten at every logger restart
            elif rtype == RECORD_DATA and topic in topics:
                name = topics[topic]
                entry = DECODERS.get(name)
                if entry is None or entry[0] != len(payload):
                    continue
                try:
                    values = entry[1](payload)
                except (ValueError, OverflowError, struct.error):
                    continue
                rows.setdefault(name, []).append({"timestamp": host_ns / 1e9, **values})

        tables = {
            name: pl.DataFrame(r).sort("timestamp").unique("timestamp", keep="last", maintain_order=True)
            for name, r in rows.items()
        }
        if not tables:
            raise ValueError(f"No decodable telemetry records in {self.path}")

        # RTK positions are (0, 0) until the fix: base the rows on RTK only if it ever got one
        rtk = tables.get("rtk_position")
        if rtk is not None and (rtk["RTK:lat_p"] != 0).any():
            base_name = "rtk_position"
        elif "gps_position" in tables:
            base_name = "gps_position"
        else:
            base_name = max(tables, key=lambda k: tables[k].height)
        df = tables[base_name]
        for name, table in tables.items():
            if name != base_name:
                df = df.join_asof(table, on="timestamp", strategy="backward")

        df = df.with_columns(
            pl.from_epoch((pl.col("timestamp") * 1e6).round().cast(pl.Int64), time_unit="us").alias("datetime")
        )
        for col in ("RTK:lat_p", "RTK:lon_p", "RTK:hmsl_p"):
            if col in df.columns:  # no fix yet -> null, not a point at (0, 0)
                df = df.with_columns(
                    pl.when(pl.col("RTK:lat_p") == 0).then(None).otherwise(pl.col(col)).alias(col)
                )
        if "flight_status" in df.columns:
            df = df.with_columns((pl.col("flight_status") == FLIGHT_STATUS_IN_AIR).alias("isflying"))

        logger.info(
            f"PSDK telemetry {self.path}: {df.height} rows on {base_name}, "
            f"topics {sorted(tables)}, {df['datetime'].min()} -> {df['datetime'].max()} UTC"
        )
        self.data = df
