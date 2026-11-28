"""Matrice 400 flight record (.txt of the remote controller's app), decoded by PILS.

The logs are written here byte by byte, as the app writes them (format 14:
XOR and AES; format 12: XOR only), with keys the test chooses; only the call
to DJI's keychain API is replaced. The decoder was also checked against
pydjirecord 1.3.0 on real Matrice 400 logs (identical records and OSD values).
"""

import base64
import json
import math
import struct
from datetime import UTC, datetime

import polars as pl
import pytest
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from pils.drones.dji.m400 import txtlog
from pils.drones.dji.m400.flightrecord import (
    DJIFlightRecordError,
    M400FlightRecord,
    resolve_api_key,
)
from pils.drones.dji.m400.txtlog import FlightLog, xor_decode, xor_key

START = datetime(2026, 11, 28, 11, 38, 29, 559000, tzinfo=UTC)
KEYS = {  # feature point -> (AES key, IV)
    txtlog.BASE: (bytes(range(32)), bytes(range(16, 32))),
    txtlog.AGRICULTURE: (bytes(range(1, 33)), bytes(range(2, 18))),
    txtlog.RC: (bytes(range(3, 35)), bytes(range(4, 20))),
    txtlog.DJI_FLY_CUSTOM: (bytes(range(5, 37)), bytes(range(6, 22))),
    txtlog.BATTERY: (bytes(range(7, 39)), bytes(range(8, 24))),
}


# --- Writing a log -----------------------------------------------------------


def xor_encode(plain: bytes, record_type: int, first: int = 0x5A) -> bytes:
    key = xor_key(first, record_type)
    return bytes([first]) + bytes(b ^ key[i % 8] for i, b in enumerate(plain))


def details_block(start=START, total_s=0.3, name="Matrice 400") -> bytes:
    d = bytearray(400)
    d[0:9] = b"Via Roma\x00"  # street address: never kept
    struct.pack_into("<iiq", d, 83, 3, 0, int(start.timestamp() * 1000))
    struct.pack_into("<ddfi", d, 99, -67.7869, -22.9597, 0.95, int(total_s * 1000))
    struct.pack_into("<3f", d, 123, 599.0, 8.6, 8.0)
    struct.pack_into("<f", d, 267, 49804.2)  # take-off altitude x10
    d[271] = 181
    d[280 : 280 + len(name)] = name.encode()
    d[312:328] = b"1581F00TEST0000A"
    d[376] = 2
    d[377:380] = bytes([17, 7, 2])
    return bytes(d)


def osd(fly_time, lat=-22.9597, lon=-67.7869, height=0.0) -> bytes:
    d = bytearray(50)
    struct.pack_into("<dd", d, 0, math.radians(lon), math.radians(lat))
    struct.pack_into("<7h", d, 16, int(height * 10), 10, -20, 0, 15, -5, 900)
    d[34] = 5 << 2  # gps level 5
    d[36] = 26  # satellites
    struct.pack_into("<H", d, 42, round(fly_time * 10))
    return bytes(d)


def home(lat=-22.9597594, lon=-67.7869144) -> bytes:
    d = bytearray(41)
    struct.pack_into("<ddf", d, 0, math.radians(lon), math.radians(lat), 49804.2)
    return bytes(d)


LAG = 0.37  # the aircraft's time minus the app's (start + fly time), s


def aircraft_time(i):
    return START.timestamp() + 0.1 * i + LAG


def record48(t, home_h=5180.2, h=5185.0) -> bytes:
    d = bytearray(71)
    struct.pack_into("<dd", d, 0, math.radians(-67.7869), math.radians(-22.9597))
    struct.pack_into("<h", d, 20, 900)  # yaw x 10
    seconds = math.floor(t)
    struct.pack_into("<II", d, 24, seconds, round((t - seconds) * 1e9))
    d[33] = 22
    struct.pack_into("<fff", d, 50, home_h, 12.5, h)
    struct.pack_into("<f", d, 66, 31.46)
    return bytes(d)


def sats(n):  # a satellite count as stored: shifted, the low bit set
    return (n << 1) | 1


def record45(
    lat=-22.95976, lon=-67.78692, h=5180.3, base=(-22.9597319, -67.7867337, 5180.63)
) -> bytes:
    d = bytearray(83)
    d[1] = 50
    d[2:11] = bytes(sats(n) for n in (10, 6, 8, 10, 6, 8, 9, 6, 7))
    struct.pack_into("<ddf", d, 11, lat, lon, h)
    struct.pack_into("<ddf", d, 31, *base)
    struct.pack_into("<3f", d, 65, 0.0022, 0.0021, 0.0051)
    d[77:80] = bytes(sats(n) for n in (7, 7, 6))  # Galileo; QZSS left 0: unknown
    return bytes(d)


def rc_display(throttle) -> bytes:
    return bytes(7) + struct.pack("<4H", 1024, 1024, throttle, 1024)


def message(text) -> bytes:
    return text.encode() + b"\x00"


def frame(i) -> list[tuple[int, bytes]]:
    return [(1, osd(0.1 * i, height=float(i))), (48, record48(aircraft_time(i)))]


DEFAULT_RECORDS = [
    (2, home()),
    *frame(0),
    (45, record45()),
    (10, message("Strong wind")),
    (10, message("Low battery")),
    *frame(1),
    (62, rc_display(1684)),
    *frame(2),
]


# Groups found on Matrice 400 logs that the decoder must know (see the test below)
APP_GROUPS = {
    254: txtlog.DJI_FLY_CUSTOM,
    253: txtlog.DJI_FLY_CUSTOM,
    57: txtlog.AIR_LINK,
}


class Writer:
    """A log of format 14 (or 12) with the records given, encrypted with KEYS."""

    def __init__(self, records=None, version=14, start=START):
        self.version = version
        self.ivs = {fp: iv for fp, (key, iv) in KEYS.items()}
        body = b""
        if version >= 13:
            for fp in KEYS:  # the encrypted keys: what the API is sent
                payload = struct.pack("<HH", fp, 4) + b"KEY" + bytes([fp])
                body += self.record(56, payload)
        for kind, data in records if records is not None else DEFAULT_RECORDS:
            body += self.record(kind, data)
        details = details_block(start)
        prefix = struct.pack("<QHB", 0, 0, version).ljust(100, b"\0")
        if version >= 13:
            info = xor_encode(
                bytes([1]) + struct.pack("<H", 400) + details + b"\0\0", 0
            )
            aux = struct.pack("<BH", 0, len(info)) + info
            aux += struct.pack("<BH", 1, 3) + struct.pack("<HB", 1, 7)
            self.raw = prefix + aux + body
        else:  # format 12: details at 100, records at 536
            self.raw = prefix + details.ljust(436, b"\0") + body

    def record(self, kind, plain) -> bytes:
        data = plain
        # the groups of the app's logs, independently of the decoder's table
        fp = APP_GROUPS.get(kind) or txtlog.feature_point(kind, self.version)
        if self.version >= 13 and fp in KEYS:
            key, _ = KEYS[fp]
            padder = padding.PKCS7(128).padder()
            padded = padder.update(plain) + padder.finalize()
            enc = Cipher(algorithms.AES(key), modes.CBC(self.ivs[fp])).encryptor()
            data = enc.update(padded) + enc.finalize()
            self.ivs[fp] = data[-16:]
            data += b"\x00"  # the byte after the AES blocks
        if self.version >= 7:
            data = xor_encode(data, kind)
        return struct.pack("<BH", kind, len(data)) + data + b"\xff"


def api_answer():
    return [
        [
            {
                "featurePoint": f"FR_Standardization_Feature_X_{fp}",
                "aesKey": base64.b64encode(key).decode(),
                "aesIv": base64.b64encode(iv).decode(),
            }
            for fp, (key, iv) in KEYS.items()
        ]
    ]


@pytest.fixture
def api(monkeypatch, tmp_path):
    """DJI's keychain API, replaced: the requests it got."""
    calls = []

    def post(body, api_key):
        calls.append((body, api_key))
        return api_answer()

    monkeypatch.setattr(txtlog, "_post_keychains", post)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("DJI_API_KEY", raising=False)
    return calls


def write(tmp_path, writer, name="DJIFlightRecord_2026-11-28_[12-38-29].txt"):
    f = tmp_path / name
    f.write_bytes(writer.raw)
    return f


# --- The container -----------------------------------------------------------


def test_xor_decoding_undoes_the_apps_encoding():
    plain = bytes(range(40))
    assert xor_decode(xor_encode(plain, 45, first=0x11), 45) == plain
    assert xor_decode(b"", 1) == b""


def test_the_header_gives_the_format_and_the_details():
    log = FlightLog.from_bytes(Writer().raw)
    assert log.version == 14 and log.department == 7 and log.keychain_version == 1
    d = log.details
    assert d.start_time == START and d.total_time == pytest.approx(0.3)
    assert d.aircraft_name == "Matrice 400" and d.aircraft_sn == "1581F00TEST0000A"
    assert d.app_version == "17.7.2" and d.product_type == 181
    assert d.take_off_altitude == pytest.approx(4980.42, abs=0.01)


def test_the_keychain_request_carries_the_logs_encrypted_keys():
    body = FlightLog.from_bytes(Writer().raw).keychain_request()
    assert body["version"] == 1 and body["department"] == 7
    (keys,) = body["keychainsArray"]
    assert {k["featurePoint"] for k in keys} >= {
        "FR_Standardization_Feature_Base_1",
        "FR_Standardization_Feature_Agriculture_4",
    }
    assert base64.b64decode(keys[0]["aesCiphertext"]) == b"KEY\x01"


def test_records_are_decrypted_with_the_keys_and_their_chained_ivs():
    log = FlightLog.from_bytes(Writer().raw)
    keys = [txtlog.Keychain.from_api(api_answer()[0])]
    records = [r for r in log.records(keys) if r.type != 56]
    assert [r.type for r in records] == [kind for kind, _ in DEFAULT_RECORDS]
    assert [r.data for r in records] == [data for _, data in DEFAULT_RECORDS]


def test_a_stream_cut_short_ends_at_the_last_whole_record():
    log = FlightLog.from_bytes(Writer().raw[:-5])  # the last record (a 48) cut
    keys = [txtlog.Keychain.from_api(api_answer()[0])]
    kinds = [r.type for r in log.records(keys) if r.type != 56]
    assert kinds == [kind for kind, _ in DEFAULT_RECORDS][:-1]


# --- The table ---------------------------------------------------------------


def test_the_frames_are_timed_by_the_aircraft(tmp_path, api):
    r = M400FlightRecord(write(tmp_path, Writer()))
    r.load_data(api_key="KEY")
    t = r.data
    assert t.height == 3
    # record 48's seconds and nanoseconds, not the app's start + fly time
    assert t["timestamp"].to_list() == pytest.approx(
        [aircraft_time(i) for i in range(3)]
    )
    assert r.details["aircraft_minus_app_s"] == pytest.approx(LAG, abs=1e-6)
    assert t["datetime"].dtype == pl.Datetime("us", "UTC")
    assert t["osd_height"].to_list() == [0.0, 1.0, 2.0]  # the file's order
    assert t["osd_latitude"][0] == pytest.approx(-22.9597)
    assert t["osd_yaw"][0] == pytest.approx(90.0) and t["osd_gps_level"][0] == 5
    assert t["osd_h_speed"][0] == pytest.approx(math.hypot(1.0, 2.0))
    # record 48: in every frame
    assert t["osd_abs_altitude"][0] == pytest.approx(5185.0)
    assert t["osd_abs_yaw"][0] == pytest.approx(90.0) and t["osd_abs_gps_num"][0] == 22
    assert t["osd_home_distance"][0] == pytest.approx(12.5)
    assert t["osd_geoid_height"][0] == pytest.approx(31.46)


def test_without_record_48_the_frames_have_the_apps_time(tmp_path, api):
    r = M400FlightRecord(write(tmp_path, Writer([(1, osd(0.0)), (1, osd(0.1))])))
    r.load_data(api_key="KEY")
    assert r.data["timestamp"].to_list() == pytest.approx(
        [START.timestamp(), START.timestamp() + 0.1]
    )
    assert r.details["aircraft_minus_app_s"] is None


def test_a_record_48_early_or_late_goes_to_its_own_frame(tmp_path, api):
    o = lambda i: (1, osd(0.1 * i, height=float(i)))  # noqa: E731
    p = lambda i: (48, record48(aircraft_time(i), h=5000.0 + i))  # noqa: E731
    records = [
        o(0),
        p(0),
        o(1),  # its 48 comes late, after the next OSD
        o(2),
        p(1),
        p(2),
        o(3),
        p(3),
        p(4),  # frame 4's 48 comes early
        o(4),
        o(5),
        p(5),
        o(6),  # none at all: interpolated
        o(7),
        p(7),
    ]
    r = M400FlightRecord(write(tmp_path, Writer(records)))
    r.load_data(api_key="KEY")
    t = r.data
    assert t["osd_abs_altitude"].to_list()[:6] == [5000.0 + i for i in range(6)]
    assert t["osd_abs_altitude"][6] is None
    assert t["timestamp"].to_list() == pytest.approx(
        [aircraft_time(i) for i in range(8)]
    )


def test_epoch_records_are_only_in_their_frame_states_hold(tmp_path, api):
    r = M400FlightRecord(write(tmp_path, Writer()))
    r.load_data(api_key="KEY")
    t = r.data
    # RTK (1 Hz): only in the frame it came in
    assert t["rtk_latitude"].to_list()[0] == pytest.approx(-22.95976)
    assert t["rtk_latitude"].to_list()[1:] == [None, None]
    assert t["rtk_pos_flag"][0] == 50
    assert t["rtk_base_height"][0] == pytest.approx(5180.63)
    assert t["rtk_std_height"][0] == pytest.approx(0.0051)
    assert (t["rtk_antenna1_gps"][0], t["rtk_base_glonass"][0]) == (10, 7)
    assert t["rtk_antenna2_galileo"][0] == 7 and t["rtk_base_qzss"][0] is None
    # the app's messages of a frame, joined
    assert t["app_warn"].to_list() == ["Strong wind; Low battery", None, None]
    # the home point (before the first frame) and the sticks hold
    assert t["home_latitude"].to_list() == pytest.approx([-22.9597594] * 3)
    assert t["rc_throttle"].to_list() == [None, 1684, 1684]


def test_a_home_point_without_a_fix_has_no_position(tmp_path, api):
    no_fix = struct.pack("<ddf", 800000.0, 800000.0, 0.0).ljust(41, b"\0")
    r = M400FlightRecord(write(tmp_path, Writer([(2, no_fix), *frame(0)])))
    r.load_data(api_key="KEY")
    assert math.isnan(r.data["home_latitude"][0])


def test_the_details_are_kept_without_the_street_address(tmp_path, api):
    r = M400FlightRecord(write(tmp_path, Writer()))
    r.load_data(api_key="KEY")
    assert r.details["version"] == 14
    assert r.details["start_time"] == START.isoformat()
    assert r.details["aircraft_sn"] == "1581F00TEST0000A"
    assert "Via Roma" not in json.dumps(r.details)


def test_a_log_without_frames_gives_an_empty_table(tmp_path, api):
    r = M400FlightRecord(write(tmp_path, Writer([(2, home())])))
    r.load_data(api_key="KEY")
    assert r.data.is_empty()


# --- The keys ----------------------------------------------------------------


def test_a_log_of_format_13_or_later_needs_the_key(tmp_path, api, monkeypatch):
    f = write(tmp_path, Writer())
    with pytest.raises(DJIFlightRecordError, match="DJI_API_KEY"):
        M400FlightRecord(f).load_data()
    monkeypatch.setenv("DJI_API_KEY", "ENVKEY")
    M400FlightRecord(f).load_data()
    assert api[-1][1] == "ENVKEY"
    assert resolve_api_key("ARG") == "ARG"


def test_the_keys_are_cached_unless_asked_not_to(tmp_path, api):
    f = write(tmp_path, Writer())
    M400FlightRecord(f).load_data(api_key="KEY")
    M400FlightRecord(f).load_data(api_key="KEY")
    assert len(api) == 1  # the second from the cache
    M400FlightRecord(f).load_data(api_key="KEY", use_cache=False)
    assert len(api) == 2
    assert list((tmp_path / "cache" / "pils" / "dji-keychains").glob("*.json"))


def test_an_older_log_needs_no_key_and_no_api_call(tmp_path, api):
    r = M400FlightRecord(write(tmp_path, Writer(version=12)))
    r.load_data()
    assert api == [] and r.data.height == 3
    assert r.data["rtk_latitude"][0] == pytest.approx(-22.95976)


def test_an_api_error_is_a_flight_record_error(tmp_path, api, monkeypatch):
    def refuse(body, api_key):
        raise txtlog.FlightLogError(
            "DJI keychain API: HTTP 403 (the DJI API key is not valid)"
        )

    monkeypatch.setattr(txtlog, "_post_keychains", refuse)
    with pytest.raises(DJIFlightRecordError, match="403"):
        M400FlightRecord(write(tmp_path, Writer())).load_data(api_key="BAD")


# --- Not a flight record -----------------------------------------------------


def test_files_that_are_not_flight_records(tmp_path):
    short = tmp_path / "short.txt"
    short.write_bytes(b"too short")
    text = tmp_path / "text.txt"
    text.write_bytes(b"x" * 200)  # format byte 0x78: unknown
    for f in (short, text):
        with pytest.raises(DJIFlightRecordError, match="not a DJI flight log"):
            M400FlightRecord(f).load_data(api_key="KEY")


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        M400FlightRecord(tmp_path / "none.txt").load_data(api_key="KEY")


# --- More records (custom, app GPS, flight controller, link, components) -----


def custom(ms, distance_km=0.5) -> bytes:
    return struct.pack("<BBffq", 0, 1, 6.3, distance_km, ms)


def test_a_record_left_undecrypted_does_not_break_its_groups_chain(tmp_path, api):
    # 254 shares the key (and the chained IV) of the app's records (5, 9, 10):
    # if it were not decrypted, the next custom record's first block would be
    # garbage (as the app's time was, before)
    t0 = int(START.timestamp() * 1000)
    records = [
        *frame(0),
        (5, custom(t0)),
        (254, bytes.fromhex("05000200020000000834")),
        (5, custom(t0 + 100, 0.6)),
        *frame(1),
    ]
    r = M400FlightRecord(write(tmp_path, Writer(records)))
    r.load_data(api_key="KEY")
    t = r.data
    # both custom records in frame 0: the second is kept, and decoded right
    assert t["app_time"][0] == pytest.approx(t0 / 1000 + 0.1)
    assert t["app_time"][1] is None
    assert t["app_distance"][0] == pytest.approx(600.0, rel=1e-6)
    assert t["app_is_video"][0] is True


def test_app_gps_flight_controller_link_and_home_relative_position(tmp_path, api):
    o = bytearray(osd(0.0))
    o += bytes(85 - len(o))
    struct.pack_into("<ff", o, 74, 12.5, -3.25)
    records = [
        (1, bytes(o)),
        (14, struct.pack("<ddf", -22.95975, -67.78687, 5181.5)),
        (19, bytes([2, 0x07]) + bytes(31)),
        (49, bytes([0x80 | 90])),
        (49, bytes([75])),
        *frame(1),
    ]
    r = M400FlightRecord(write(tmp_path, Writer(records)))
    r.load_data(api_key="KEY")
    t = r.data
    assert (t["osd_north"][0], t["osd_east"][0]) == (12.5, -3.25)
    assert t["app_gps_latitude"][0] == pytest.approx(-22.95975)
    assert t["app_gps_altitude"].to_list() == pytest.approx([5181.5] * 2)  # holds
    assert t["mc_fail_safe"][0] == 2 and t["mc_is_obstacle_avoidance_on"][0]
    assert (t["link_uplink_signal"][0], t["link_downlink_signal"][0]) == (90, 75)


def test_firmware_and_component_serials_go_to_the_details(tmp_path, api):
    serial = b"1581F00TEST0000A2XYZ"
    records = [
        (15, bytes([3, 0, 9, 5, 0, 32]) + bytes(12)),
        (40, struct.pack("<HB", 2, len(serial) + 1) + serial + b"\0"),
        *frame(0),
    ]
    r = M400FlightRecord(write(tmp_path, Writer(records)))
    r.load_data(api_key="KEY")
    assert r.details["firmware"] == {"flight controller": "9.5.0.32"}
    assert r.details["component_serials"] == {"aircraft": serial.decode()}


def test_rtk_without_a_base_and_the_flight_controllers_estimates(tmp_path, api):
    fc = struct.pack("<HHHHBB", 0, 1509, 0xFFFF, 30, 20, 9)
    records = [*frame(0), (45, record45(base=(0.0, 0.0, 0.0))), (63, fc), *frame(1)]
    r = M400FlightRecord(write(tmp_path, Writer(records)))
    r.load_data(api_key="KEY")
    t = r.data
    assert t["rtk_base_latitude"][0] is None and t["rtk_base_height"][0] is None
    assert t["fc_flight_time_left"].to_list() == [1509, 1509]  # holds
    assert t["fc_go_home_time"][0] is None  # 0xFFFF: none yet
    assert (
        t["fc_land_time"][0],
        t["fc_go_home_battery"][0],
        t["fc_land_battery"][0],
    ) == (30, 20, 9)
