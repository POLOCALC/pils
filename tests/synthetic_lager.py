"""Writers of LAGER's logs, to test the decoders: the byte layouts of the
Matrice 600 OSDK frames, the Matrice 400 psdk_telemetry stream and the gimbal
pickles, written here independently of the decoders."""

import binascii
import math
import pickle
import struct

from pils.drones.dji.m600 import protocol as m600


# --- Matrice 600 -----------------------------------------------------------------------------------------------
def broadcast_payload(
    *,
    t_ticks=400,
    q=(1.0, 0.0, 0.0, 0.0),
    lat_deg=45.5,
    lon_deg=9.2,
    alt_m=210.5,
    hgt_m=30.0,
    gps_lat=455000000,
    gps_lon=92000000,
    rtk_lat=45.5001,
    rtk_lon=9.2001,
    rtk_yaw=1234,
    status=True,
) -> bytes:
    """A full broadcast payload (flag 0x0FFF, 267 bytes with the status)."""
    size = m600.OFF_STATUS + (4 if status else 0)
    b = bytearray(size)
    struct.pack_into("<H", b, m600.OFF_FLAG, 0x0FFF if status else 0x07FF)
    struct.pack_into("<I", b, m600.OFF_TMS, t_ticks)
    struct.pack_into("<I", b, m600.OFF_TNS, 5)
    struct.pack_into("<4f", b, m600.OFF_Q, *q)
    struct.pack_into("<3f", b, m600.OFF_A, 0.0, 0.0, -1.0)
    struct.pack_into("<3f", b, m600.OFF_V, 1.0, 2.0, -0.5)
    b[m600.OFF_VI] = 3
    struct.pack_into("<3f", b, m600.OFF_W, 0.01, 0.02, 0.03)
    struct.pack_into("<d", b, m600.OFF_LAT, math.radians(lat_deg))
    struct.pack_into("<d", b, m600.OFF_LON, math.radians(lon_deg))
    struct.pack_into("<f", b, m600.OFF_ALT, alt_m)
    struct.pack_into("<f", b, m600.OFF_HGT, hgt_m)
    b[m600.OFF_GPSH] = 4
    struct.pack_into("<6f", b, m600.OFF_RP, 1, 2, 3, 4, 5, 6)
    struct.pack_into("<II", b, m600.OFF_GPS_DATE, 20261119, 101500)
    struct.pack_into("<3i", b, m600.OFF_GPS_LON, gps_lon, gps_lat, 210500)
    struct.pack_into("<3f", b, m600.OFF_GPS_VEL, 10.0, 20.0, 30.0)
    struct.pack_into("<2f", b, m600.OFF_GPS_DTL, 0.8, 1.1)
    struct.pack_into("<H", b, m600.OFF_GPS_DTL + 32, 17)
    struct.pack_into("<2d", b, m600.OFF_RTK + 8, rtk_lon, rtk_lat)
    struct.pack_into("<f", b, m600.OFF_RTK + 24, 211.0)
    struct.pack_into("<h", b, m600.OFF_RTK + 40, rtk_yaw)
    b[m600.OFF_RTK + 42] = 50
    b[m600.OFF_RTK + 43] = 16
    struct.pack_into("<3h", b, m600.OFF_MX, 1, 2, 3)
    struct.pack_into("<6h", b, m600.OFF_RC, 10, 20, 30, 40, 50, 60)
    struct.pack_into("<3f", b, m600.OFF_GIMBAL, 0.5, -45.0, 90.0)
    b[m600.OFF_GIMBAL + 12] = 7
    if status:
        struct.pack_into("<4B", b, m600.OFF_STATUS, 3, 6, 1, 0)
    return bytes(b)


def battery_payload(percent=77) -> bytes:
    b = bytearray(17)
    struct.pack_into("<H", b, 0, 0x3000)
    struct.pack_into("<IiiB", b, 2, 5000, 22000, -3000, percent)
    struct.pack_into("<BB", b, 15, 2, 1)
    return bytes(b)


def osdk_frame(
    payload: bytes, *, cmd=(0x02, 0x00), ack=False, seq=1, corrupt=False
) -> bytes:
    body = payload if ack else bytes(cmd) + payload
    length = m600.HEADER_LEN + len(body) + m600.CRC32_LEN
    head = bytearray(m600.HEADER_LEN)
    head[0] = 0xAA
    struct.pack_into("<I", head, 0, 0xAA | (length << 8) | ((1 << 29) if ack else 0))
    struct.pack_into("<H", head, 8, seq)
    raw = bytes(head) + body
    crc = m600.crc32(raw) ^ (1 if corrupt else 0)
    return raw + struct.pack("<I", crc)


def m600_file(frames: list[tuple[float, bytes]], truncate: int = 0) -> bytes:
    """``[(host time, frame)]`` in LAGER's file layout; ``truncate`` cuts that many bytes off the end."""
    data = b"".join(struct.pack("<dI", t, len(f)) + f for t, f in frames)
    return data[:-truncate] if truncate else data


# --- Matrice 400 -----------------------------------------------------------------------------------------------
SYNC = b"\xa5\x5a"


def psdk_record(
    rtype: int, topic: int, payload: bytes, host_ns: int, ac_ms: int = 0
) -> bytes:
    body = (
        struct.pack("<BBHHqII", 1, rtype, topic, len(payload), host_ns, ac_ms, 0)
        + payload
    )
    return SYNC + body + struct.pack("<H", binascii.crc_hqx(body, 0xFFFF))


def topic_table(topics: dict[int, tuple[str, int, int]]) -> bytes:
    """``{id: (name, hz, size)}``"""
    out = struct.pack("<H", len(topics))
    for tid, (name, hz, size) in topics.items():
        out += struct.pack("<HHHB", tid, hz, size, 1) + name.encode().ljust(32, b"\0")
    return out


def psdk_status(state: int, text: str) -> bytes:
    return bytes([state]) + text.encode()


# --- Gimbal ----------------------------------------------------------------------------------------------------
def gimbal_file(chunks: list[list[dict]]) -> bytes:
    return b"".join(pickle.dumps(c) for c in chunks)


def gimbal_entry(t: float, keyword: str, **data) -> dict:
    return {"timestamp": t, "keyword": keyword, "data": data}


def m400_file(truncate: int = 0, garbage: bool = False, bad_crc: bool = False) -> bytes:
    """A short Matrice 400 recording: a status, the topic table, then quaternion, fused position, battery,
    an unknown topic and a payload of the wrong size."""
    topics = {
        1: ("quaternion", 10, 16),
        2: ("position_fused", 10, 24),
        3: ("battery1", 1, 33),
        4: ("not_a_topic", 1, 2),
        5: ("gps_position", 5, 12),
    }
    q = struct.pack("<4f", math.cos(math.radians(45)), 0, 0, math.sin(math.radians(45)))
    fused = struct.pack(
        "<ddfHBB", math.radians(9.2), math.radians(45.5), 210.0, 20, 1, 2
    )
    battery = struct.pack(
        "<BBiiIIhBB8sBBB",
        0,
        1,
        24000,
        -2500,
        9000,
        8000,
        305,
        6,
        88,
        b"\0" * 8,
        0,
        0,
        90,
    )
    gps = struct.pack("<3i", 92000000, 455000000, 210500)
    out = psdk_record(2, 0, psdk_status(4, "subscribed"), 1_000_000_000)
    out += psdk_record(3, 0, topic_table(topics), 1_000_000_001)
    out += psdk_record(1, 3, battery, 1_100_000_000, 100)  # slow topic first
    out += psdk_record(1, 1, q, 1_200_000_000, 200)
    out += psdk_record(1, 4, b"\x00\x00", 1_250_000_000)  # unknown topic
    out += psdk_record(1, 5, gps[:8], 1_260_000_000)  # wrong size
    out += psdk_record(1, 5, gps, 1_270_000_000, 270)
    if garbage:
        out += b"\x00\xa5\x00junk\xa5\x5a\xff"
    rec = psdk_record(1, 2, fused, 1_300_000_000, 300)
    if bad_crc:
        rec = rec[:-1] + bytes([rec[-1] ^ 0xFF])
    out += rec
    out += psdk_record(1, 1, q, 1_400_000_000, 400)
    return out[:-truncate] if truncate else out
