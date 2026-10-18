"""Payload files of the older campaigns and damaged recordings.

2022-2023 pickles, the April 2024 ADC records, log time formats, corrupted
u-blox data, cut CSV files, and the drone GPS as the sync reference when
the payload GPS is missing or does not cover the flight.
"""

import datetime
import pickle
import struct

import numpy as np
import polars as pl
import pytest

from pils.flight import DroneData, Flight, PayloadData, _time_coverage
from pils.sensors import legacy
from pils.sensors.adc import ADC, decode_adc_file_struct, detect_struct_layout
from pils.sensors.gps import GPS, read_ubx
from pils.sensors.inclinometer import (
    Inclinometer,
    KernelInclinometer,
    read_csv_tolerant,
)
from pils.utils.tools import (
    get_logpath_from_datapath,
    parse_log_datetime,
    read_log_time,
)

T0 = 1_651_240_000.0  # 2022-04-29


# -- logs -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            "2025/12/08 19:25:43.590  [Thread-1]  INFO:Sensor ZED-F9P started",
            datetime.datetime(2025, 12, 8, 19, 25, 43, 590000),
        ),
        (
            "2024/04/29 18:59:31  [Thread-2]  INFO:Sensor ZED-F9P started",
            datetime.datetime(2024, 4, 29, 18, 59, 31),
        ),
        (
            "02/21/2023 19:07:09    [Thread-1]    INFO:Sensor ZED-F9R started",
            datetime.datetime(2023, 2, 21, 19, 7, 9),
        ),
        ("no time here", None),
    ],
)
def test_parse_log_datetime(line, expected):
    assert parse_log_datetime(line) == expected


def test_read_log_time_without_fraction(tmp_path):
    log = tmp_path / "x_file.log"
    log.write_text("2024/04/29 18:59:31  [Thread-2]  INFO:Sensor ZED-F9P started\n")
    start, date = read_log_time("Sensor ZED-F9", log)
    assert start == datetime.datetime(2024, 4, 29, 18, 59, 31)
    assert date == datetime.date(2024, 4, 29)


def test_logpath_in_the_sensors_folder(tmp_path):
    sensors = tmp_path / "aux" / "sensors"
    sensors.mkdir(parents=True)
    data = sensors / "20230221_150900_ADC.pck"
    data.write_bytes(b"")
    log = sensors / "20230221_150900_log.log"
    log.write_text("")
    assert get_logpath_from_datapath(data) == log


# -- 2022 and 2023 pickles ----------------------------------------------------


def pickles(path, values, padding=64):
    with open(path, "wb") as f:
        for value in values:
            pickle.dump(value, f, protocol=4)
        f.write(b"\x00" * padding)  # preallocated
    return path


def test_read_pickles_skips_padding_and_refuses_classes(tmp_path):
    path = pickles(tmp_path / "a.pck", [["ADC", [T0, 0.1]], None, b"xy"])
    with open(path, "ab") as f:
        pickle.dump(datetime.datetime(2022, 1, 1), f, protocol=4)  # a class
    assert list(legacy.read_pickles(path)) == [["ADC", [T0, 0.1]], None, b"xy"]


def test_undo_clock_jumps():
    # Saved clock, then set by the network 3 days later
    times = np.r_[np.arange(5) * 0.02, 259_200 + np.arange(5) * 0.02]
    fixed = legacy.undo_clock_jumps(times)
    assert np.allclose(np.diff(fixed), 0.02)
    assert fixed[-1] == times[-1]


def test_2022_adc_and_inclinometer(tmp_path):
    t = T0 + np.arange(200) * 0.025
    t[:50] -= 250_000  # before the clock was set
    pickles(
        tmp_path / "20220429_140200_ADC_1.pck",
        [["ADC", [float(x), -0.08]] for x in t],
    )
    pickles(
        tmp_path / "20220429_140200_INC_1.pck",
        [["INC", [float(x), (-0.5, 45.0)]] for x in t],
    )
    adc = legacy.read_legacy_adc(tmp_path)
    assert adc.height == 200
    assert adc["amplitude"][0] == pytest.approx(-80.0)  # mV
    assert np.allclose(np.diff(adc["timestamp"].to_numpy()), 0.025)
    assert adc["timestamp"][-1] == pytest.approx(t[-1])

    inc = legacy.read_legacy_inclinometer(tmp_path)
    assert inc.columns == ["timestamp", "pitch", "roll", "datetime"]
    assert inc["pitch"][0] == 0.5 and inc["roll"][0] == 45.0


def test_2023_adc_blocks(tmp_path):
    # 150 Hz for 600 s; the times stored as float32 (128 s steps)
    true = 1_677_002_900.0 + np.arange(90_000) / 150.0
    samples = np.empty(true.size, dtype=[("t", "<f4"), ("v", "<i4")])
    samples["t"] = true.astype(np.float32)
    samples["v"] = 16
    raw = samples.tobytes()
    pickles(
        tmp_path / "20230221_150900_ADC.pck",
        [raw[i : i + 256] for i in range(0, len(raw), 256)],
    )
    adc = legacy.read_legacy_adc(tmp_path)
    assert adc.height == true.size
    assert np.max(np.abs(adc["timestamp"].to_numpy() - true)) < 0.05
    assert adc["amplitude"][0] == pytest.approx(16 * 4.096 / 32768 * 1e3)


def test_2023_empty_inclinometer(tmp_path):
    pickles(tmp_path / "20230221_150900_INC.pck", [None] * 100, padding=0)
    assert legacy.read_legacy_inclinometer(tmp_path).is_empty()
    inc = Inclinometer(tmp_path)
    assert inc.sensor_type == "legacy"
    inc.load_data()
    assert inc.data.is_empty()


def test_no_legacy_files(tmp_path):
    assert legacy.read_legacy_adc(tmp_path) is None
    assert legacy.read_legacy_inclinometer(tmp_path) is None


# -- ADC binary layouts ----------------------------------------------------------


def test_adc_struct_cut_record(tmp_path):
    path = tmp_path / "x_ADC.bin"
    records = b"".join(
        struct.pack("<dqf", T0 + i * 0.01, 500_000, -140.625) for i in range(100)
    )
    path.write_bytes(records + b"\x01\x02\x03" + b"\x00" * 20)  # cut, zeros
    assert detect_struct_layout(path.read_bytes()) == "dqf"
    data = decode_adc_file_struct(path)
    assert data.height == 100
    assert data["amplitude"][0] == pytest.approx(-140.625)


def test_adc_april_2024_records(tmp_path):
    path = tmp_path / "20240429_151148_ADC.bin"
    path.write_bytes(
        b"".join(struct.pack("<df", T0 + i * 0.025, -0.046) for i in range(300))
    )
    assert detect_struct_layout(path.read_bytes()) == "df"
    adc = ADC(tmp_path, logpath="none")
    adc.load_data()
    assert adc.data.height == 300
    assert "reading_time" not in adc.data.columns
    assert adc.data["amplitude"][0] == pytest.approx(-46.0, abs=1e-3)  # mV


def test_adc_pickles_through_the_sensor(tmp_path):
    pickles(
        tmp_path / "20220429_140200_ADC_1.pck",
        [["ADC", [T0 + i, 0.001]] for i in range(10)],
    )
    adc = ADC(tmp_path, logpath="none")
    adc.load_data()
    assert adc.data.height == 10


def test_adc_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="No ADC file"):
        ADC(tmp_path)


# -- GPS ----------------------------------------------------------------------


def test_gps_missing_and_ubx_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="No payload GPS file"):
        GPS(tmp_path)
    (tmp_path / "20251128_171906_GPS.ubx").write_bytes(b"")
    assert GPS(tmp_path).data_path.name == "20251128_171906_GPS.ubx"


def test_read_ubx_skips_corrupted_bytes(tmp_path):
    from pyubx2 import UBXMessage

    message = UBXMessage("NAV", "NAV-CLOCK", 0, iTOW=1000, clkB=1, clkD=2)
    path = tmp_path / "g_GPS.bin"
    # "$" then binary: pyubx2 reads an NMEA sentence and fails to decode it
    path.write_bytes(message.serialize() + b"$\xb5\xfe\x00,\r\n" + message.serialize())
    assert [m.identity for m in read_ubx(path)] == ["NAV-CLOCK", "NAV-CLOCK"]


# -- inclinometer files -----------------------------------------------------------


def test_read_csv_tolerant(tmp_path):
    path = tmp_path / "x_INC_imu.csv"
    path.write_bytes(
        b"timestamp_ns,acc_X_m_s2,acc_Y_m_s2\n"
        b"1000,0.1,-0.2\n"
        b"2000,0.2,-0.3\n"
        b"\x00\x00\x00\n"
        b"3000,0.3,-"  # cut by a power loss
    )
    frame = read_csv_tolerant(path)
    assert frame["timestamp_ns"].to_list() == [1000, 2000]


def test_kernel_without_messages(tmp_path):
    path = tmp_path / "x_INC.bin"
    path.write_bytes(b"\x00" * 1024)
    with pytest.raises(ValueError, match="No Kernel inclinometer messages"):
        KernelInclinometer(path, logpath="none").load_data()


# -- the drone GPS as the reference ---------------------------------------------------


def flight_with(drone=True, gps_span=None):
    flight = Flight(
        {
            "drone_data_folder_path": "/tmp/x/drone",
            "aux_data_folder_path": "/tmp/x/aux",
        }
    )
    t = np.arange(0.0, 300.0, 0.2)
    flight.raw_data.payload_data = PayloadData()
    flight.raw_data.payload_data.adc = pl.DataFrame(
        {"timestamp": np.arange(0.0, 300.0, 0.01), "amplitude_mv": 1.0}
    )
    if gps_span is not None:
        g = t[(t >= gps_span[0]) & (t <= gps_span[1])]
        flight.raw_data.payload_data.gps = pl.DataFrame(
            {
                "timestamp": g,
                "lat_deg": -22.96 + 1e-5 * np.sin(g / 20),
                "lon_deg": -67.78 + 1e-5 * np.cos(g / 30),
                "h_ell_m": 5200 + np.sin(g / 10),
                "h_msl_m": 5160 + np.sin(g / 10),
            }
        )
    if drone:
        flight.raw_data.drone_data = DroneData(
            drone_df=pl.DataFrame(
                {
                    "timestamp": t,
                    "lat_deg": -22.96 + 1e-5 * np.sin(t / 20),
                    "lon_deg": -67.78 + 1e-5 * np.cos(t / 30),
                    "h_msl_m": 5160 + np.sin(t / 10),
                }
            ),
            litchi_df=None,
        )
    return flight


def test_sync_to_the_drone_without_payload_gps():
    flight = flight_with()
    synced = flight.sync(target_rate={"drone": 5.0, "payload": 10.0})
    assert flight.sync_reference == "drone_gps"
    assert "reference_gps" not in synced
    assert {"drone", "adc"} <= set(synced)
    assert flight.sync_offset_details["drone_gps"]["method"] == "reference"
    assert flight.sync_offset_details["payload_clock"]["method"] == "none"


def test_payload_gps_that_misses_the_flight():
    flight = flight_with(gps_span=(0.0, 30.0))  # 10 % of the drone track
    flight.sync(target_rate={"drone": 5.0, "payload": 10.0})
    assert flight.sync_reference == "drone_gps"


def test_payload_gps_is_the_reference():
    flight = flight_with(gps_span=(0.0, 300.0))
    synced = flight.sync(target_rate={"drone": 5.0, "payload": 10.0})
    assert flight.sync_reference == "payload_gps"
    assert "reference_gps" in synced


def test_time_coverage():
    ref = pl.Series([0.0, 100.0])
    assert _time_coverage(pl.Series([50.0, 300.0]), ref) == 0.5
    assert _time_coverage(pl.Series([200.0, 300.0]), ref) == 0.0


def test_positions_at_zero_are_dropped():
    flight = flight_with(gps_span=(0.0, 300.0))
    drone = flight.raw_data.drone_data.drone
    # every 5th drone row without a solution, as DJI RTK rows
    flight.raw_data.drone_data.drone = drone.with_columns(
        [
            pl.when(pl.int_range(pl.len()) % 5 == 0)
            .then(0.0)
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("lat_deg", "lon_deg", "h_msl_m")
        ]
    )
    flight.sync(target_rate={"drone": 5.0, "payload": 10.0})
    assert flight.sync_reference == "payload_gps"
    assert flight.sync_offset_details["drone_gps"]["correlation"] > 0.9


def test_payload_gps_epochs_without_a_fix_are_dropped():
    flight = flight_with(gps_span=(0.0, 300.0))
    gps = flight.raw_data.payload_data.gps
    # every 4th epoch without a fix: a position far off, a huge accuracy
    no_fix = pl.int_range(pl.len()) % 4 == 0
    flight.raw_data.payload_data.gps = gps.with_columns(
        pl.when(no_fix).then(-44.9).otherwise(pl.col("lat_deg")).alias("lat_deg"),
        pl.when(no_fix).then(4.3e6).otherwise(5.0).alias("h_acc_m"),
    )
    flight.sync(target_rate={"drone": 5.0, "payload": 10.0})
    assert flight.sync_reference == "payload_gps"
    assert flight.sync_offset_details["drone_gps"]["correlation"] > 0.9
