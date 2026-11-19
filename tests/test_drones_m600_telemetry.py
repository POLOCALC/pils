"""Matrice 600 OSDK telemetry (LAGER's uav_telemetry.bin)."""

import math

import pytest

from pils.drones.dji.m600 import protocol
from pils.drones.dji.m600.telemetry import FILENAME, M600Telemetry

from .synthetic_lager import battery_payload, broadcast_payload, m600_file, osdk_frame


def test_crc32_is_djis_variant():
    # init 0x3AA3 and no final xor: not the zlib CRC
    assert protocol.crc32(b"123456789") == 3839482900
    assert len(protocol.CRC32_TAB) == 256 and protocol.CRC32_TAB[1] == 0x77073096


def test_decodes_a_full_broadcast_frame(tmp_path):
    f = tmp_path / FILENAME
    f.write_bytes(
        m600_file(
            [(1000.0, osdk_frame(broadcast_payload(t_ticks=800, q=(1, 0, 0, 0))))]
        )
    )
    d = M600Telemetry(tmp_path)
    d.load_data()
    row = d.data.row(0, named=True)
    assert d.data.height == 1 and row["timestamp"] == 1000.0
    assert row["t_ticks"] == 800 and row["flag"] == 0x0FFF
    assert row["lat_deg"] == pytest.approx(45.5) and row["lon_deg"] == pytest.approx(
        9.2
    )
    assert row["alt_m"] == pytest.approx(210.5) and row["hgt_m"] == pytest.approx(30.0)
    assert (row["roll"], row["pitch"], row["yaw"]) == pytest.approx((0, 0, 0))
    assert (row["q0"], row["q1"], row["q2"], row["q3"]) == (1.0, 0.0, 0.0, 0.0)
    assert row["gps_lat_deg"] == pytest.approx(45.5) and row[
        "gps_hfsl_m"
    ] == pytest.approx(210.5)
    assert row["rtk_lat_deg"] == pytest.approx(45.5001) and row[
        "rtk_yaw_deg"
    ] == pytest.approx(123.4)
    assert row["rtk_pos_health"] == 50 and row["gps_nsv"] == 17
    assert (row["gim_roll"], row["gim_pitch"], row["gim_yaw"]) == pytest.approx(
        (0.5, -45.0, 90.0)
    )
    assert row["flight_status"] == 3 and row["gps_date"] == 20261119
    assert d.stats["frames"] == 1 and d.stats["bad_crc"] == 0


def test_a_yaw_quaternion_gives_the_euler_angles():
    q = (
        math.cos(math.radians(45)),
        0.0,
        0.0,
        math.sin(math.radians(45)),
    )  # 90 deg about z
    assert protocol.quat_to_euler(*q) == pytest.approx((0.0, 0.0, 90.0))


def test_battery_frames_fill_their_columns_only_unless_filled(tmp_path):
    f = tmp_path / FILENAME
    f.write_bytes(
        m600_file(
            [
                (1.0, osdk_frame(broadcast_payload())),
                (2.0, osdk_frame(battery_payload(77))),
            ]
        )
    )
    d = M600Telemetry(f)
    d.load_data()
    assert d.data.height == 2
    assert d.data["bat_percentage"].to_list() == [None, 77]
    assert d.data["lat_deg"].to_list()[1] is None
    d.load_data(fill=True)
    assert d.data["lat_deg"].to_list()[1] == pytest.approx(45.5)
    assert d.data["bat_percentage"].to_list() == [None, 77]


def test_frames_that_are_not_telemetry_or_corrupt_are_left_out(tmp_path):
    frames = [
        (1.0, osdk_frame(broadcast_payload())),
        (2.0, osdk_frame(b"\x00\x00", ack=True)),
        (3.0, osdk_frame(broadcast_payload(), cmd=(0x00, 0x00))),
        (4.0, osdk_frame(broadcast_payload(), corrupt=True)),
        (5.0, osdk_frame(broadcast_payload())),
    ]
    (tmp_path / FILENAME).write_bytes(m600_file(frames))
    d = M600Telemetry(tmp_path)
    d.load_data()
    assert d.data["timestamp"].to_list() == [1.0, 5.0]
    assert (d.stats["ack"], d.stats["other_command"], d.stats["bad_crc"]) == (1, 1, 1)


def test_a_file_cut_in_the_middle_of_a_frame_keeps_the_frames_before(tmp_path):
    frames = [(float(i), osdk_frame(broadcast_payload(t_ticks=i))) for i in range(3)]
    (tmp_path / FILENAME).write_bytes(m600_file(frames, truncate=10))
    d = M600Telemetry(tmp_path)
    d.load_data()
    assert d.data["t_ticks"].to_list() == [0, 1]
    assert d.stats["truncated"] == 1


def test_an_impossible_frame_length_stops_the_read(tmp_path):
    good = m600_file([(1.0, osdk_frame(broadcast_payload()))])
    (tmp_path / FILENAME).write_bytes(
        good + b"\x00" * 8 + b"\x00\x00\x00\x00" + b"\xff" * 40
    )
    d = M600Telemetry(tmp_path)
    d.load_data()
    assert d.data.height == 1 and d.stats["short"] == 1


def test_rows_are_sorted_by_time(tmp_path):
    frames = [
        (3.0, osdk_frame(broadcast_payload(t_ticks=3))),
        (1.0, osdk_frame(broadcast_payload(t_ticks=1))),
    ]
    (tmp_path / FILENAME).write_bytes(m600_file(frames))
    d = M600Telemetry(tmp_path)
    d.load_data()
    assert d.data["timestamp"].to_list() == [1.0, 3.0]


def test_missing_file_and_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        M600Telemetry(tmp_path).load_data()
    (tmp_path / FILENAME).write_bytes(b"")
    d = M600Telemetry(tmp_path)
    d.load_data()
    assert d.data.is_empty()
