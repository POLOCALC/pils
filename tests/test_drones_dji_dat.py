"""Tests for the DJI .dat record scanner and decoders."""

import math
import struct
from datetime import UTC, datetime

import numpy as np
import polars as pl
import pytest

from pils.drones.DJIDrone import (
    DAT_RECORDS,
    DJIDrone,
    add_euler_angles,
    calc_checksum,
    decode_records,
    scan_records,
    unwrap_ticks,
)


def record(msg_type: int, tick: int, payload: bytes) -> bytes:
    """A .dat record: header, payload XOR-ed with the tick's low byte, CRC."""
    key = tick & 0xFF
    length = 10 + len(payload) + 2
    body = (
        bytes([0x55, length, 0, 0])
        + struct.pack("<HI", msg_type, tick)
        + bytes(b ^ key for b in payload)
    )
    return body + struct.pack("<H", calc_checksum(body, 0, len(body)))


def gps_payload(second: int, lon: float = -67.78, lat: float = -22.96) -> bytes:
    payload = bytearray(72)
    struct.pack_into("<II", payload, 0, 20251208, 192700 + second)
    struct.pack_into("<ii", payload, 8, round(lon * 1e7), round(lat * 1e7))
    struct.pack_into("<i", payload, 16, 5300_000)
    struct.pack_into("<H", payload, 64, 15)
    return bytes(payload)


def imu_payload(quat=(1.0, 0.0, 0.0, 0.0), temperature=45.5) -> bytes:
    payload = bytearray(120)
    struct.pack_into("<dd", payload, 0, math.radians(-67.78), math.radians(-22.96))
    struct.pack_into("<fff", payload, 20, 0.1, -0.2, -1.0)  # accel (g)
    struct.pack_into("<f", payload, 32, math.radians(10.0))  # gyroX (rad/s)
    struct.pack_into("<ffff", payload, 48, *quat)
    struct.pack_into("<h", payload, 106, round(temperature * 100))
    payload[116] = 12  # numSats
    return bytes(payload)


class TestScan:
    def test_finds_records_with_0x55_inside(self):
        # The XOR key is the tick's low byte: a payload byte equal to
        # 0x55 ^ key is written as 0x55 and used to split the record
        tick = 0x1000_0042
        payload = bytes([0x55 ^ 0x42] * 20)
        data = b"\x00\x55\x01garbage" + record(2048, tick, payload) + b"\x55"
        data += record(2096, tick + 1, gps_payload(0))
        arr, starts, lengths = scan_records(data)
        assert starts.size == 2
        assert lengths.tolist() == [32, 84]
        assert arr[starts[0] + 10 : starts[0] + 30].tolist() == [0x55] * 20

    def test_drops_bad_checksum(self):
        good = record(2096, 100, gps_payload(0))
        bad = bytearray(record(2096, 200, gps_payload(1)))
        bad[20] ^= 0xFF
        _, starts, _ = scan_records(bytes(bad) + good)
        assert starts.tolist() == [len(bad)]

    def test_empty(self):
        _, starts, _ = scan_records(b"")
        assert starts.size == 0


class TestUnwrap:
    def test_wrap_and_out_of_order_records(self):
        ticks = np.array([4_294_000_000, 4_294_900_000, 1_000, 4_294_950_000, 5_000])
        assert unwrap_ticks(ticks).tolist() == [
            4_294_000_000,
            4_294_900_000,
            2**32 + 1_000,
            4_294_950_000,  # written after the wrap, ticked before it
            2**32 + 5_000,
        ]

    def test_small_backward_steps_are_not_resets(self):
        ticks = np.array([23_587_534, 23_584_069, 23_600_000])
        assert unwrap_ticks(ticks).tolist() == ticks.tolist()

    def test_reset(self):
        ticks = np.array([1_500_000_000, 1_000, 2_000])
        assert unwrap_ticks(ticks).tolist() == [
            1_500_000_000,
            1_500_001_000,
            1_500_002_000,
        ]


class TestDecode:
    def test_imu_fields_and_scaling(self):
        data = record(2048, 1_000, imu_payload())
        arr, starts, lengths = scan_records(data)
        df = decode_records(
            arr,
            starts,
            lengths,
            np.array([1_000]),
            "IMU_ATTI(0)",
            DAT_RECORDS[2048]["fields"],
        )
        row = df.row(0, named=True)
        assert row["IMU_ATTI(0):longitude"] == pytest.approx(-67.78)
        assert row["IMU_ATTI(0):latitude"] == pytest.approx(-22.96)
        assert row["IMU_ATTI(0):accelZ"] == pytest.approx(-1.0)
        assert row["IMU_ATTI(0):gyroX"] == pytest.approx(10.0, rel=1e-6)
        assert row["IMU_ATTI(0):temperature"] == pytest.approx(45.5)
        assert row["IMU_ATTI(0):numSats"] == 12
        assert df.schema["IMU_ATTI(0):numSats"] == pl.Int64
        assert row["tick"] == 1_000

    def test_short_records_are_skipped(self):
        data = record(2048, 1, imu_payload()[:50])
        arr, starts, lengths = scan_records(data)
        df = decode_records(
            arr, starts, lengths, np.array([1]), "IMU", DAT_RECORDS[2048]["fields"]
        )
        assert df.height == 0

    def test_motor_scaling(self):
        payload = bytearray(185)
        struct.pack_into("<hhhhh", payload, 1, 1234, 3800, 495, 41, 1920)
        struct.pack_into("<H", payload, 19, 960)
        arr, starts, lengths = scan_records(record(10090, 7, bytes(payload)))
        df = decode_records(
            arr, starts, lengths, np.array([7]), "Motor", DAT_RECORDS[10090]["fields"]
        )
        row = df.row(0, named=True)
        assert row["Motor:Current:RFront"] == pytest.approx(12.34)
        assert row["Motor:Speed:RFront"] == 3800
        assert row["Motor:Volts:RFront"] == pytest.approx(49.5)
        assert row["Motor:EscTemp:RFront"] == 41
        assert row["Motor:PPMrecv:RFront"] == pytest.approx(100.0)
        assert row["Motor:PPMsend:RFront"] == pytest.approx(50.0)
        assert "Motor:Speed:RSide" in df.columns


class TestEuler:
    def angles(self, quat):
        df = pl.DataFrame(
            {f"Q:quat{c}": [v] for c, v in zip("WXYZ", quat, strict=True)}
        )
        row = add_euler_angles(df, "Q").row(0, named=True)
        return row["Q:roll"], row["Q:pitch"], row["Q:yaw"]

    def test_identity(self):
        assert self.angles((1.0, 0.0, 0.0, 0.0)) == pytest.approx((0, 0, 0))

    def test_yaw(self):
        half = math.radians(30) / 2
        roll, pitch, yaw = self.angles((math.cos(half), 0, 0, math.sin(half)))
        assert (roll, pitch, yaw) == pytest.approx((0, 0, 30))

    def test_datcon_roll_is_rotation_about_x(self):
        half = math.radians(20) / 2
        roll, pitch, yaw = self.angles((math.cos(half), math.sin(half), 0, 0))
        assert (roll, pitch, yaw) == pytest.approx((20, 0, 0))

    def test_gimbal_lock_branch(self):
        s = math.sqrt(0.5)
        roll, pitch, yaw = self.angles((s, s, 0.0, 0.0))  # test = 0.5
        assert (roll, pitch) == pytest.approx((0, 90))


class TestLoadDat:
    @pytest.fixture
    def dat_file(self, tmp_path):
        data = b"BUILD header \x55\x00 padding"
        tick = 4_294_000_000  # wraps during the file
        step = 2_222_222  # ~0.5 s of 4.5 MHz ticks
        for second in range(30):
            for half in range(2):
                t = (tick + (2 * second + half) * step) % 2**32
                data += record(2048, t, imu_payload(temperature=40 + second))
            t = (tick + 2 * second * step) % 2**32
            data += record(2096, t + 7, gps_payload(second, lon=-67.78 + second * 1e-5))
            data += record(53234, t + 9, bytes(72))
        path = tmp_path / "flight_drone.dat"
        path.write_bytes(data)
        return path

    def test_gps_rtk_and_records(self, dat_file):
        drone = DJIDrone(dat_file)
        drone._load_from_dat()

        gps = drone.data["GPS"]
        assert gps.height == 20  # last 10 dropped, as before
        assert gps.columns[0] == "msg_type"
        assert gps["datetime"][0] == datetime(2025, 12, 8, 19, 27, tzinfo=UTC)
        assert gps.schema["datetime"] == pl.Datetime("us", "UTC")
        assert gps["timestamp"][1] - gps["timestamp"][0] == 1.0
        assert (gps["tick"].diff().drop_nulls() > 0).all()  # unwrapped
        # RTK has no valid date: no datetime columns, as before
        assert "timestamp" not in drone.data["RTK"].columns

        imu = drone.records["IMU_ATTI(0)"]
        assert imu.height == 60
        assert {"IMU_ATTI(0):roll", "IMU_ATTI(0):yaw"} <= set(imu.columns)
        assert (imu["tick"].diff().drop_nulls() > 0).all()
        assert set(drone.records) == {"IMU_ATTI(0)"}

    def test_timestamps_from_gps_fit(self, dat_file):
        drone = DJIDrone(dat_file)
        drone._load_from_dat()
        drone.tick_to_time = (1 / 4.5e6, 1_765_000_000.0)
        drone._timestamp_records()
        imu = drone.records["IMU_ATTI(0)"]
        expected = imu["tick"].to_numpy() / 4.5e6 + 1_765_000_000.0
        np.testing.assert_allclose(imu["timestamp"].to_numpy(), expected)

    def test_no_fit_no_timestamps(self, dat_file):
        drone = DJIDrone(dat_file)
        drone._load_from_dat()
        drone._timestamp_records()
        assert "timestamp" not in drone.records["IMU_ATTI(0)"].columns


class TestReceivers:
    def test_rtk_table_and_gps_record(self, tmp_path):
        """The drone table is the RTK solution; the main GPS is a record."""
        data = b""
        tick = 1_000_000
        for second in range(30):
            t = tick + second * 4_500_000
            data += record(2096, t, gps_payload(second, lon=-67.78 + second * 1e-5))
            rtk = bytearray(72)
            struct.pack_into("<dd", rtk, 8, -67.78, -22.96 + second * 1e-6)
            struct.pack_into("<h", rtk, 52, 359 if second % 2 else 0)
            data += record(53234, t + 2_250_000, bytes(rtk))
        path = tmp_path / "flight_drone.dat"
        path.write_bytes(data)

        drone = DJIDrone(path)
        drone.load_data(use_dat=True)
        assert set(drone.data["msg_type"].unique()) == {53234}
        assert drone.data.height == 20  # 30 records, last 10 dropped
        # Its own rows, not interpolated: yaw is 0 or 359 only
        assert set(drone.data["RTK:yaw"].unique()) <= {0, 359}
        # Half a second after each GPS record, from the tick-time fit
        gps = drone.records["GPS"]
        np.testing.assert_allclose(
            drone.data["timestamp"][:3].to_numpy(),
            gps["timestamp"][:3].to_numpy() + 0.5,
            atol=1e-6,
        )
        assert drone.data.schema["datetime"] == pl.Datetime("us", "UTC")
