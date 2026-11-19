"""Gremsy T7 gimbal telemetry (LAGER's gimbal_telemetry.bin)."""

import os
import pickle

import pytest

from pils.gimbals.gremsy_t7 import FILENAME, GremsyT7Telemetry

from .synthetic_lager import gimbal_entry, gimbal_file


def chunks():
    return [
        [
            gimbal_entry(
                10.0,
                "MOUNT_ORIENTATION",
                roll=0.5,
                pitch=-30.0,
                yaw=120.0,
                yaw_absolute=121.0,
            ),
            gimbal_entry(10.1, "RAW_IMU", xgyro=1, ygyro=2, zgyro=3),
        ],
        [
            gimbal_entry(
                9.9, "GIMBAL_DEVICE_ATTITUDE_STATUS", q=[1.0, 0.0, 0.0, 0.0], flags=12
            ),
            gimbal_entry(
                10.2,
                "MOUNT_ORIENTATION",
                roll=0.6,
                pitch=-31.0,
                yaw=120.5,
                yaw_absolute=121.5,
            ),
        ],
    ]


def load(tmp_path, content):
    (tmp_path / FILENAME).write_bytes(content)
    g = GremsyT7Telemetry(tmp_path)
    g.load_data()
    return g


def test_messages_become_rows_sorted_by_time(tmp_path):
    g = load(tmp_path, gimbal_file(chunks()))
    assert g.data["keyword"].to_list() == [
        "GIMBAL_DEVICE_ATTITUDE_STATUS",
        "MOUNT_ORIENTATION",
        "RAW_IMU",
        "MOUNT_ORIENTATION",
    ]
    assert g.data["timestamp"].to_list() == [9.9, 10.0, 10.1, 10.2]
    assert g.data["pitch"].to_list() == [None, -30.0, None, -31.0]
    assert g.data["q"].to_list()[0] == [1.0, 0.0, 0.0, 0.0]
    assert g.complete


def test_one_table_per_message_with_only_its_fields(tmp_path):
    g = load(tmp_path, gimbal_file(chunks()))
    assert set(g.records) == {
        "MOUNT_ORIENTATION",
        "RAW_IMU",
        "GIMBAL_DEVICE_ATTITUDE_STATUS",
    }
    assert g.records["MOUNT_ORIENTATION"].columns == [
        "timestamp",
        "keyword",
        "roll",
        "pitch",
        "yaw",
        "yaw_absolute",
    ]
    assert (
        g.records["RAW_IMU"].height == 1 and "pitch" not in g.records["RAW_IMU"].columns
    )


def test_integer_and_float_values_of_one_field_share_a_column(tmp_path):
    g = load(
        tmp_path,
        gimbal_file([[gimbal_entry(1.0, "A", v=1), gimbal_entry(2.0, "A", v=2.5)]]),
    )
    assert g.data["v"].to_list() == [1.0, 2.5]


def test_a_partial_last_write_keeps_what_precedes_it(tmp_path):
    content = gimbal_file(chunks())
    g = load(tmp_path, content[:-9])
    assert not g.complete and g.data.height == 2


def test_a_file_that_builds_anything_but_plain_data_is_refused(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("touch " + str(tmp_path / "pwned"),))

    content = gimbal_file(chunks()[:1]) + pickle.dumps([Evil()])
    g = load(tmp_path, content)
    assert not (tmp_path / "pwned").exists()
    assert not g.complete and g.data.height == 2


def test_entries_without_a_timestamp_are_skipped(tmp_path):
    g = load(
        tmp_path,
        gimbal_file(
            [[{"keyword": "X", "data": {}}, gimbal_entry(1.0, "RAW_IMU", xgyro=1)]]
        ),
    )
    assert g.data.height == 1


def test_missing_and_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        GremsyT7Telemetry(tmp_path).load_data()
    g = load(tmp_path, b"")
    assert g.data.is_empty() and g.records == {}
