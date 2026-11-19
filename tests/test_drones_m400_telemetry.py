"""Matrice 400 psdk_telemetry stream (LAGER's uav_telemetry.bin)."""

import pytest

from pils.drones.dji.m400 import decoders, stream
from pils.drones.dji.m400.parameters import TELEMETRY_FILENAME
from pils.drones.dji.m400.telemetry import M400Telemetry

from .synthetic_lager import m400_file, psdk_record


def load(tmp_path, content, **kw):
    (tmp_path / TELEMETRY_FILENAME).write_bytes(content)
    d = M400Telemetry(tmp_path)
    d.load_data(**kw)
    return d


def test_rows_carry_their_topic_and_forward_fill(tmp_path):
    d = load(tmp_path, m400_file())
    assert d.data["topic"].to_list() == [
        "battery1",
        "quaternion",
        "gps_position",
        "position_fused",
        "quaternion",
    ]
    assert d.data["timestamp"].to_list() == pytest.approx([1.1, 1.2, 1.27, 1.3, 1.4])
    assert d.data["t_aircraft_ms"].to_list() == [100, 200, 270, 300, 400]
    last = d.data.row(-1, named=True)
    assert last["lat_deg"] == pytest.approx(45.5) and last["lon_deg"] == pytest.approx(
        9.2
    )  # carried forward
    assert last["alt_m"] == pytest.approx(210.0) and last["fused_nsv"] == 20
    assert last["yaw"] == pytest.approx(90.0) and last["roll"] == pytest.approx(0.0)
    assert last["gps_lat_deg"] == pytest.approx(45.5) and last[
        "gps_hfsl_m"
    ] == pytest.approx(210.5)
    assert last["bat1_voltage_mv"] == 24000 and last["bat1_percentage"] == 88
    assert last["bat1_temp_c"] == pytest.approx(30.5)
    # the M600's battery keys, from the single battery
    assert last["bat_percentage"] == 88 and last["bat_capacity_mah"] == 8000
    assert "q" not in d.data.columns


def test_without_fill_a_row_has_only_its_topic(tmp_path):
    d = load(tmp_path, m400_file(), fill=False)
    first = d.data.row(0, named=True)
    assert first["bat1_percentage"] == 88 and first["yaw"] is None
    assert (
        "bat_percentage" not in d.data.columns
    )  # the summary is made on the filled table


def test_status_unknown_topics_and_wrong_sizes_are_counted(tmp_path):
    d = load(tmp_path, m400_file())
    assert d.status == [(1.0, "subscribed", "subscribed")]
    assert (
        d.stats["unknown_topic"] == 1
        and d.stats["bad_size"] == 1
        and d.stats["bad_crc"] == 0
    )


def test_garbage_between_records_is_skipped(tmp_path):
    clean = load(tmp_path, m400_file())
    noisy = load(tmp_path, m400_file(garbage=True))
    assert noisy.data.equals(clean.data)


def test_a_record_with_a_bad_crc_is_dropped_and_counted(tmp_path):
    d = load(tmp_path, m400_file(bad_crc=True))
    assert "position_fused" not in d.data["topic"].to_list() and d.stats["bad_crc"] == 1


def test_a_file_cut_in_the_middle_keeps_the_records_before(tmp_path):
    d = load(tmp_path, m400_file(truncate=5))
    assert d.data["topic"].to_list()[-1] == "position_fused"


def test_the_stream_parser_takes_chunks_of_any_size():
    data = m400_file()
    p = stream.StreamParser()
    out = []
    for i in range(0, len(data), 7):
        out += p.feed(data[i : i + 7])
    assert len(out) == len(stream.StreamParser().feed(data)) == 9 and p.bad_crc == 0


def test_values_that_cannot_be_decoded_do_not_stop_the_telemetry(monkeypatch):
    import struct

    errors = {}
    nan_fix = struct.pack("<6f2I2H", 1, 1, float("nan"), 1, 1, 1, 1, 1, 1, 1)
    assert (
        decoders.decode("gps_details", nan_fix, errors)["gps_fix"] is None
        and errors == {}
    )

    def boom(_payload):
        raise ValueError("cannot convert float NaN to integer")

    monkeypatch.setitem(decoders.DECODERS, "altitude_fused", (4, boom))
    assert decoders.decode("altitude_fused", b"\x00" * 4, errors) == {} and errors == {
        "bad_value": 1
    }


def test_a_record_before_any_topic_table_is_ignored(tmp_path):
    d = load(tmp_path, psdk_record(1, 1, b"\x00" * 16, 1))
    assert d.data.is_empty()


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        M400Telemetry(tmp_path).load_data()
