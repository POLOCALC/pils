"""The container of a DJI app flight log (``DJIFlightRecord_*.txt``): header, records, decryption.

PILS's own reader of the logs the DJI apps write (DJI Pilot 2 on the Matrice
400's remote controller): the header and its details, the stream of records
and their decryption, and the request of the decryption keys to DJI's API.
What the records hold is read by :mod:`.records`; the table of a flight by
:mod:`.flightrecord`.

The layout follows DJI's FlightRecordParsingLib (C++) and the MIT-licensed
``dji-log-parser`` (Rust, L. Vauvillier) and ``pydjirecord`` (Python,
A. Rembish), which documented it.

Layout
------
::

    prefix (100 bytes)    u64 detail offset, u16 detail length, u8 format version
    format >= 13          auxiliary blocks at byte 100: u8 kind, u16 size, data
                          kind 0 (info): XOR-encoded; u8, u16 n, n bytes of
                          details, u16 m, m bytes of signature
                          kind 1 (version): u16 version, u8 department
                          the records follow them (prefix offset 0) or start at
                          the prefix offset
    format < 13           details at the prefix offset (format < 12) or byte 100
    records               u8 type, u16 n, n bytes, 0xFF; until the end of the
                          file (format >= 12) or the details

Decryption: from format 7 every record is XOR-encoded with a key derived
(CRC-64) from its first byte and its type; from format 13 most are also
AES-256-CBC encrypted, with a key and IV per "feature point" (the group of the
record type, :func:`feature_point`), the IV chained from record to record.
The keys are in the log, encrypted (key-storage records, type 56); DJI's
keychain API returns them in clear for a DJI Open API app key
(:func:`fetch_keychains`). A key-storage-recover record (type 50) starts the
next set of keys.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PREFIX_SIZE = 100
KEYCHAIN_API = "https://dev.dji.com/openapi/v1/flight-records/keychains"
API_TIMEOUT_S = 30.0
CACHE_DAYS = 30
MIN_VERSION, MAX_VERSION = 6, 14  # the formats read here

KEY_STORAGE = 56
KEY_STORAGE_RECOVER = 50
_END_OF_RECORD = 0xFF


class FlightLogError(RuntimeError):
    """The log cannot be read (not a DJI flight log, cut short, no keys...)."""


# --- XOR (format 7+) ---------------------------------------------------------

_CRC64_POLY = 0x95AC9329AC4BC9B5  # Jones, bit-reversed
_MASK64 = (1 << 64) - 1
_XOR_MAGIC = 0x123456789ABCDEF0


def _crc64_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = (crc >> 1) ^ _CRC64_POLY if crc & 1 else crc >> 1
        table.append(crc)
    return table


_CRC64 = _crc64_table()


def crc64(seed: int, data: bytes) -> int:
    """CRC-64 (Jones polynomial, reflected) of ``data`` from ``seed``."""
    crc = seed & _MASK64
    for byte in data:
        crc = _CRC64[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc


def xor_key(first_byte: int, record_type: int) -> bytes:
    """Return the 8-byte XOR key of a record whose (encoded) data starts with ``first_byte``."""
    seed = (first_byte + record_type) & 0xFF
    return crc64(
        seed, ((_XOR_MAGIC * first_byte) & _MASK64).to_bytes(8, "little")
    ).to_bytes(8, "little")


def xor_decode(data: bytes, record_type: int) -> bytes:
    """Decode a record's data: its first byte gives the key and is dropped."""
    if not data:
        return data
    key = xor_key(data[0], record_type)
    body = data[1:]
    reps = key * (len(body) // 8 + 1)
    return bytes(b ^ k for b, k in zip(body, reps, strict=False))


# --- AES (format 13+) --------------------------------------------------------

# Feature points: the groups of record types that share a key
BASE, VISION, WAYPOINT, AGRICULTURE, AIR_LINK, AFTER_SALES = 1, 2, 3, 4, 5, 6
DJI_FLY_CUSTOM, PLAINTEXT, FLIGHT_HUB, GIMBAL, RC, CAMERA = 7, 8, 9, 10, 11, 12
BATTERY, FLY_SAFE, SECURITY = 13, 14, 15
FEATURE_POINT_NAMES = {
    BASE: "Base",
    VISION: "Vision",
    WAYPOINT: "Waypoint",
    AGRICULTURE: "Agriculture",
    AIR_LINK: "AirLink",
    AFTER_SALES: "AfterSales",
    DJI_FLY_CUSTOM: "DJIFlyCustom",
    PLAINTEXT: "Plaintext",
    FLIGHT_HUB: "FlightHub",
    GIMBAL: "Gimbal",
    RC: "RC",
    CAMERA: "Camera",
    BATTERY: "Battery",
    FLY_SAFE: "FlySafe",
    SECURITY: "Security",
}


def feature_point(record_type: int, version: int) -> int:
    """Return the feature point (key group) of a record type."""
    if record_type in (KEY_STORAGE, KEY_STORAGE_RECOVER):
        return PLAINTEXT
    if record_type == 55:
        return SECURITY
    if record_type in (17, 18):
        return VISION
    if record_type in (31, 32, 34, 35, 36, 38, 39):
        return WAYPOINT
    if record_type in (21, 41, 43, 44, 45, 46, 47, 48):
        return AGRICULTURE
    if record_type in (49, 57):  # 57: found on Matrice 400 logs (format 14)
        return AIR_LINK
    if record_type == 62:
        return RC
    # 253, 254: found on Matrice 400 logs (format 14); while they were left
    # undecrypted, the IV chain of this group broke (record 5's first block)
    if record_type in (5, 9, 10, 20, 24, 30, 54, 253, 254):
        return DJI_FLY_CUSTOM
    if version >= 14:  # format 14 has groups of its own
        if record_type == 3:
            return GIMBAL
        if record_type in (4, 11, 29, 33):
            return RC
        if record_type == 25:
            return CAMERA
        if record_type in (7, 8, 22):
            return BATTERY
        if record_type in (28, 51, 52):
            return FLY_SAFE
    if record_type == 53:
        return FLIGHT_HUB
    if record_type in (12, 16, 19, 22, 23, 26, 27, 28, 51, 52, 93, 102, 103, 113):
        return AFTER_SALES
    if record_type in (1, 2, 3, 4, 6, 7, 8, 11, 13, 14, 15, 25, 29, 33, 40, 58, 59, 63):
        return BASE
    return PLAINTEXT


def aes_cbc_decrypt(data: bytes, key: bytes, iv: bytes) -> bytes:
    """AES-256-CBC decryption; the PKCS#7 padding removed when it is valid."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).decryptor()
    out = decryptor.update(data) + decryptor.finalize()
    pad = out[-1] if out else 0
    if 1 <= pad <= 16 and out[-pad:] == bytes([pad]) * pad:
        return out[:-pad]
    return out


class Keychain:
    """The AES key and current IV of each feature point, for one set of records."""

    def __init__(self, entries: dict[int, tuple[bytes, bytes]] | None = None) -> None:
        self._keys = dict(entries or {})

    @classmethod
    def from_api(cls, group: list[dict[str, str]]) -> Keychain:
        """From one group of the API's answer: ``featurePoint``, ``aesKey``, ``aesIv`` (base64)."""
        entries = {}
        for item in group:
            try:
                fp = int(str(item["featurePoint"]).rsplit("_", 1)[-1])
                key = base64.b64decode(item["aesKey"])
                iv = base64.b64decode(item["aesIv"])
            except (KeyError, ValueError):
                continue
            if len(key) == 32 and len(iv) == 16:
                entries[fp] = (key, iv)
        return cls(entries)

    def decrypt(self, data: bytes, fp: int) -> bytes:
        """Decrypt a record of feature point ``fp`` (XOR-decoded, its last byte dropped)."""
        if fp not in self._keys:
            return data
        content = data[:-1]
        if len(content) < 16 or len(content) % 16:
            return data
        key, iv = self._keys[fp]
        self._keys[fp] = (key, content[-16:])  # the next record's IV
        return aes_cbc_decrypt(content, key, iv)


# --- Header ------------------------------------------------------------------


def _string(raw: bytes) -> str:
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()


@dataclass
class Details:
    """The log's details (its header): what the app wrote about the flight.

    ``start_time`` is the log's start (UTC); ``total_time`` its duration (s);
    ``take_off_altitude`` as the app wrote it (barometric on the Matrice 400:
    not a height of known reference).
    """

    start_time: datetime | None
    total_time: float
    total_distance: float
    max_height: float
    max_horizontal_speed: float
    max_vertical_speed: float
    latitude: float
    longitude: float
    take_off_altitude: float
    product_type: int
    aircraft_name: str
    aircraft_sn: str
    camera_sn: str
    rc_sn: str
    battery_sn: str
    app_platform: int
    app_version: str
    record_line_count: int

    @classmethod
    def parse(cls, data: bytes) -> Details:
        """Read the details block (format 6 and later)."""
        data = data.ljust(400, b"\x00")
        o = 80 + 3  # sub-street, street, city, area (20 bytes each, not kept); 3 flags
        record_line_count, _checksum, start_ms = struct.unpack_from("<iiq", data, o)
        o += 16
        lon, lat, distance_km, total_ms = struct.unpack_from("<ddfi", data, o)
        o += 24
        max_h, max_hs, max_vs = struct.unpack_from("<3f", data, o)
        o += 12
        o += 4 + 8  # photos, video time
        o += 16 + 16 + 32 + 32  # moment pictures: sizes and positions
        o += 8 + 16  # analysis offset, user id
        (take_off,) = struct.unpack_from("<f", data, o)
        product_type = data[o + 4]
        o += 4 + 1 + 8  # + activation time
        name = _string(data[o : o + 32])
        o += 32
        sns = [_string(data[o + 16 * i : o + 16 * (i + 1)]) for i in range(4)]
        o += 64
        platform = data[o]
        app_version = ".".join(str(b) for b in data[o + 1 : o + 4])
        start = (
            datetime.fromtimestamp(start_ms / 1000, tz=UTC) if start_ms > 0 else None
        )
        return cls(
            start_time=start,
            total_time=total_ms / 1000.0,
            total_distance=distance_km * 1000.0,
            max_height=max_h,
            max_horizontal_speed=max_hs,
            max_vertical_speed=max_vs,
            latitude=lat,
            longitude=lon,
            take_off_altitude=take_off / 10.0,
            product_type=product_type,
            aircraft_name=name,
            aircraft_sn=sns[0],
            camera_sn=sns[1],
            rc_sn=sns[2],
            battery_sn=sns[3],
            app_platform=platform,
            app_version=app_version,
            record_line_count=record_line_count,
        )


@dataclass
class Record:
    """One record of the stream: its type and its data, decoded."""

    type: int
    data: bytes


@dataclass
class FlightLog:
    """A DJI app flight log: format version, details and the place of its records.

    Use :meth:`read` (or :meth:`from_bytes`); :meth:`records` decodes the
    records, with the keys of :meth:`keychain_request` for format 13 and later.
    """

    raw: bytes
    version: int
    details: Details
    records_start: int
    records_end: int
    keychain_version: int = 0
    department: int = 3  # DJI Fly, when the log does not say
    _request: dict[str, Any] | None = field(default=None, repr=False)

    @classmethod
    def read(cls, path: str | Path) -> FlightLog:
        """Read the log at ``path``."""
        return cls.from_bytes(Path(path).read_bytes())

    @classmethod
    def from_bytes(cls, raw: bytes) -> FlightLog:
        """Read the header of a log; the records are decoded by :meth:`records`."""
        if len(raw) < PREFIX_SIZE:
            raise FlightLogError("not a DJI flight log: shorter than its header")
        detail_offset, _length, version = struct.unpack_from("<QHB", raw, 0)
        if not MIN_VERSION <= version <= MAX_VERSION:
            raise FlightLogError(
                f"not a DJI flight log of a known format (version {version}; "
                f"{MIN_VERSION} to {MAX_VERSION} are read)"
            )
        keychain_version, department = 0, 3
        if version < 13:
            start = detail_offset if version < 12 else PREFIX_SIZE
            details = Details.parse(raw[start : start + 400])
            records_start = PREFIX_SIZE if version < 12 else PREFIX_SIZE + 436
            records_end = detail_offset if version < 12 else len(raw)
        else:
            pos = PREFIX_SIZE
            blocks: dict[int, bytes] = {}
            for _ in range(2):  # info, then version
                try:
                    kind, size = struct.unpack_from("<BH", raw, pos)
                except struct.error as e:
                    raise FlightLogError(
                        "the log's auxiliary blocks are cut short"
                    ) from e
                blocks[kind] = raw[pos + 3 : pos + 3 + size]
                pos += 3 + size
            if 0 not in blocks:
                raise FlightLogError("the log has no details (auxiliary info block)")
            info = xor_decode(blocks[0], 0)
            (n,) = struct.unpack_from("<H", info, 1)
            details = Details.parse(info[3 : 3 + n])
            if 1 in blocks and len(blocks[1]) >= 3:
                keychain_version, department = struct.unpack_from("<HB", blocks[1], 0)
                if department not in range(1, 9):
                    department = 3
            records_start = detail_offset or pos
            records_end = len(raw)
        return cls(
            raw=raw,
            version=version,
            details=details,
            records_start=records_start,
            records_end=records_end,
            keychain_version=keychain_version,
            department=department,
        )

    def _stream(self) -> Iterator[tuple[int, bytes]]:
        """Return the records' types and data, as stored (encoded)."""
        pos, raw = self.records_start, self.raw
        while pos + 3 <= self.records_end:
            kind = raw[pos]
            (size,) = struct.unpack_from("<H", raw, pos + 1)
            end = pos + 3 + size
            if end >= len(raw) or raw[end] != _END_OF_RECORD:
                return  # cut short, or not a record: the stream ends
            yield kind, raw[pos + 3 : end]
            pos = end + 1

    def keychain_request(self) -> dict[str, Any]:
        """Return the body of the keychain API request: the log's encrypted keys, by set."""
        if self._request is None:
            sets: list[list[dict[str, str]]] = [[]]
            for kind, data in self._stream():
                if kind == KEY_STORAGE:
                    plain = xor_decode(data, kind)
                    if len(plain) < 4:
                        continue
                    fp, n = struct.unpack_from("<HH", plain, 0)
                    sets[-1].append(
                        {
                            "featurePoint": f"FR_Standardization_Feature_"
                            f"{FEATURE_POINT_NAMES.get(fp, 'Unknown')}_{fp}",
                            "aesCiphertext": base64.b64encode(
                                plain[4 : 4 + n]
                            ).decode(),
                        }
                    )
                elif kind == KEY_STORAGE_RECOVER:
                    sets.append([])
            self._request = {
                "version": self.keychain_version,
                "department": self.department,
                "keychainsArray": sets,
            }
        return self._request

    def records(self, keychains: list[Keychain] | None = None) -> Iterator[Record]:
        """Decode the records, in the order of the file.

        Parameters
        ----------
        keychains : list of Keychain
            For format 13 and later: the sets of keys (:func:`fetch_keychains`),
            in order; without them the encrypted records stay encrypted.
        """
        sets = list(keychains or [])
        keychain = sets.pop(0) if sets else Keychain()
        for kind, data in self._stream():
            if self.version >= 7:
                data = xor_decode(data, kind)
            if self.version >= 13:
                fp = feature_point(kind, self.version)
                if fp != PLAINTEXT:
                    data = keychain.decrypt(data, fp)
            yield Record(kind, data)
            if kind == KEY_STORAGE_RECOVER:
                keychain = sets.pop(0) if sets else Keychain()


# --- Keys --------------------------------------------------------------------


def _cache_dir() -> Path:
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "pils" / "dji-keychains"


def fetch_keychains(
    log: FlightLog, api_key: str, use_cache: bool = True
) -> list[Keychain]:
    """Return the keys of a log of format 13 or later, from DJI's keychain API.

    The request carries the log's encrypted keys and the app key, nothing of
    the flight. The answer is cached for :data:`CACHE_DAYS` days under
    ``$XDG_CACHE_HOME/pils/dji-keychains`` (by the request's hash), so a log
    can be read again offline in that time.

    Raises
    ------
    FlightLogError
        If the request fails or DJI refuses it.
    """
    if log.version < 13:
        return []
    body = log.keychain_request()
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    cached = _cache_dir() / f"{digest}.json"
    data = None
    if use_cache and cached.is_file():
        if time.time() - cached.stat().st_mtime < CACHE_DAYS * 86400:
            try:
                data = json.loads(cached.read_text())
            except (OSError, ValueError):
                data = None
    if data is None:
        data = _post_keychains(body, api_key)
        if use_cache:
            try:
                cached.parent.mkdir(parents=True, exist_ok=True)
                cached.write_text(json.dumps(data))
            except OSError:
                pass
    if not isinstance(data, list):
        raise FlightLogError(f"unexpected answer of DJI's keychain API: {data!r:.200}")
    return [Keychain.from_api(group) for group in data if isinstance(group, list)]


def _post_keychains(body: dict[str, Any], api_key: str) -> Any:
    request = urllib.request.Request(
        KEYCHAIN_API,
        data=json.dumps(body).encode(),
        headers={"Api-Key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_S) as answer:
            result = json.loads(answer.read())
    except urllib.error.HTTPError as e:
        hint = " (the DJI API key is not valid)" if e.code == 403 else ""
        raise FlightLogError(f"DJI keychain API: HTTP {e.code}{hint}") from e
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        raise FlightLogError(f"DJI keychain API not reached: {e}") from e
    status = result.get("result") or {}
    if status.get("code", 0) != 0:
        raise FlightLogError(f"DJI keychain API: {status.get('msg', 'error')}")
    return result.get("data") or []
