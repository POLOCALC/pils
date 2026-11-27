"""The psdk_telemetry stream: records with a sync word, a header, a payload and a CRC-16.

Ported (copied and adapted, not imported) from LAGER, POLOCALC/lager at 825164a:
``lager/UAVs/DJI_M400/stream.py``, unchanged but for the import and
without ``parse_stats`` (pils reads no STATS record).
"""

import binascii
import struct

from . import parameters as params

HEADER = struct.Struct(params.HEADER_FMT)


def crc16(data) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF), the same as the C program."""
    return binascii.crc_hqx(data, 0xFFFF)


class StreamParser:
    """Incremental parser of the stream.

    feed() it chunks of any size (from a pipe or a file) and it returns
    the complete records found so far. Partial records stay in the buffer until the next feed();
    corrupted bytes are skipped by searching for the next sync word.
    """

    def __init__(self):
        self.buf = bytearray()
        self.bad_crc = 0  # records discarded for a CRC mismatch

    def feed(self, chunk) -> list:
        """Return [(rtype, topic, host_ns, aircraft_ms, aircraft_us, payload), ...]."""
        self.buf += chunk
        buf = self.buf
        out = []
        pos = 0
        while True:
            start = buf.find(params.SYNC, pos)
            if start < 0:
                pos = max(pos, len(buf) - 1)  # keep a possible half sync word
                break
            if len(buf) - start < params.HEADER_SIZE:
                pos = start  # header not complete yet
                break
            version, rtype, topic, length, host_ns, ac_ms, ac_us = HEADER.unpack_from(
                buf, start + 2
            )
            if version != params.STREAM_VERSION or length > params.MAX_PAYLOAD:
                pos = start + 1  # not a real header
                continue
            end = start + params.HEADER_SIZE + length + params.CRC_SIZE
            if len(buf) < end:
                pos = start  # record not complete yet
                break
            (crc,) = struct.unpack_from("<H", buf, end - params.CRC_SIZE)
            if crc16(buf[start + 2 : end - params.CRC_SIZE]) != crc:
                self.bad_crc += 1
                pos = start + 1
                continue
            out.append(
                (
                    rtype,
                    topic,
                    host_ns,
                    ac_ms,
                    ac_us,
                    bytes(buf[start + params.HEADER_SIZE : end - params.CRC_SIZE]),
                )
            )
            pos = end
        del buf[:pos]
        return out


def parse_status(payload: bytes) -> tuple:
    """STATUS record -> (state, text)."""
    return payload[0], payload[1:].decode("utf-8", "replace")


def parse_topic_table(payload: bytes) -> dict:
    """TOPIC_TABLE record -> {topic_id: {"name", "hz", "size", "subscribed"}}."""
    (count,) = struct.unpack_from("<H", payload, 0)
    count = min(
        count, (len(payload) - 2) // params.TOPIC_ENTRY_SIZE
    )  # robust to a truncated record
    topics = {}
    for i in range(count):
        off = 2 + i * params.TOPIC_ENTRY_SIZE
        tid, hz, size, subscribed = struct.unpack_from("<HHHB", payload, off)
        name = payload[off + 7 : off + params.TOPIC_ENTRY_SIZE].split(b"\0")[0].decode()
        topics[tid] = {
            "name": name,
            "hz": hz,
            "size": size,
            "subscribed": bool(subscribed),
        }
    return topics
