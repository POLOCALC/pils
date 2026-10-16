"""Tests for write-once Parquet revisions (pils.store) and Flight.save/load."""

import json
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from pils import store
from pils.flight import DroneData, Flight


@pytest.fixture
def tables():
    return {
        "drone": pl.DataFrame(
            {
                "timestamp": [0.0, 1.0],
                "rtk_lat_deg": [1.0, None],
                "rtk_num_sv": pl.Series([12, None], dtype=pl.Int64),
                "datetime": pl.Series(
                    [datetime(2025, 12, 8, 19, 27), None]
                ).dt.replace_time_zone("UTC"),
            }
        ),
        "payload": {"inclinometer": {"ins": pl.DataFrame({"roll_deg": [0.5]})}},
    }


def test_round_trip_types_nulls_and_nesting(tmp_path, tables):
    path = store.save_version(tmp_path, "rev_20260101_000000", tables, {"a": 1})
    loaded, metadata = store.load_version(tmp_path, "rev_20260101_000000")
    assert metadata == {"a": 1}
    assert loaded["drone"].equals(tables["drone"])
    assert loaded["drone"].schema == tables["drone"].schema
    assert loaded["payload"]["inclinometer"]["ins"].equals(
        tables["payload"]["inclinometer"]["ins"]
    )
    assert (path / "payload" / "inclinometer" / "ins.parquet").is_file()


def test_manifest_units_and_code_version(tmp_path, tables):
    path = store.save_version(tmp_path, "rev_1", tables, {}, kind="test")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["kind"] == "test"
    assert manifest["schema_version"] == 1
    assert set(manifest["pils"]) == {"version", "git_commit", "git_dirty"}
    columns = manifest["tables"]["drone"]["columns"]
    assert columns["rtk_lat_deg"] == {"dtype": "Float64", "unit": "deg"}
    assert "unit" not in columns["rtk_num_sv"]
    assert store.read_units(path / "drone.parquet") == {
        "timestamp": "s",
        "rtk_lat_deg": "deg",
    }
    # Parquet is readable without pils
    assert pl.read_parquet(path / "drone.parquet").equals(tables["drone"])


def test_write_once(tmp_path, tables):
    store.save_version(tmp_path, "rev_1", tables, {"first": True})
    with pytest.raises(FileExistsError):
        store.save_version(tmp_path, "rev_1", tables, {"first": False})
    assert store.load_version(tmp_path, "rev_1")[1] == {"first": True}


def test_run_files_next_to_tables_and_listing(tmp_path, tables):
    folder = tmp_path / "SATP1" / "rev_2"
    folder.mkdir(parents=True)
    (folder / "plot.png").write_bytes(b"png")
    (tmp_path / "SATP1" / "rev_1").mkdir()  # no manifest: not a revision
    store.save_version(tmp_path, "rev_2", tables, {}, parent="SATP1")
    assert (folder / "plot.png").exists()
    assert store.list_versions(tmp_path, parent="SATP1") == ["rev_2"]
    assert store.list_versions(tmp_path / "missing") == []
    with pytest.raises(KeyError):
        store.load_version(tmp_path, "rev_1", parent="SATP1")


def test_failed_write_leaves_nothing(tmp_path):
    with pytest.raises(TypeError):
        store.save_version(tmp_path, "rev_1", {"bad": object()}, {})
    assert list(tmp_path.iterdir()) == []


def test_replace(tmp_path, tables):
    store.write_tables(tmp_path / "raw", tables, replace=True)
    store.write_tables(
        tmp_path / "raw", {"drone": tables["drone"].head(1)}, replace=True
    )
    loaded, _ = store.read_tables(tmp_path / "raw")
    assert set(loaded) == {"drone"} and loaded["drone"].height == 1
    assert [p.name for p in tmp_path.iterdir()] == ["raw"]


def test_flight_save_and_load(tmp_path):
    flight = Flight({"drone_data_folder_path": str(tmp_path / "drone")})
    flight._Flight__drone_model = "dji"
    flight.raw_data.drone_data = DroneData(
        pl.DataFrame({"timestamp": [0.0], "rtk_lat_deg": [1.0]}),
        None,
        {"imu_atti_0": pl.DataFrame({"timestamp": [0.0], "yaw_deg": [3.0]})},
    )
    flight.raw_data.payload_data.gps = pl.DataFrame({"timestamp": [0.0]})
    flight.raw_data.payload_data.inclinometer = {
        "ins": pl.DataFrame({"timestamp": [0.0], "pitch_deg": [1.0]})
    }
    flight.sync_data = {"drone": pl.DataFrame({"timestamp": np.arange(3.0)})}
    flight.sync_offsets = {"drone_gps": 0.2}

    first = flight.save(tmp_path / "out", {"run": 1}, revision="rev_20260101_000000")
    second = flight.save(tmp_path / "out", {"run": 2})
    assert first != second

    loaded = Flight.load(tmp_path / "out", sync_revision=first)
    assert loaded._Flight__drone_model == "dji"
    assert loaded.raw_data.drone_data.records["imu_atti_0"]["yaw_deg"][0] == 3.0
    assert loaded.raw_data.payload_data.inclinometer["ins"].height == 1
    assert loaded.sync_data["drone"].height == 3
    assert loaded.sync_offsets == {"drone_gps": 0.2}
    manifest = store.read_manifest(tmp_path / "out" / "sync" / first)
    raw = store.read_manifest(tmp_path / "out" / "raw")
    assert manifest["metadata"]["run"] == 1
    # The raw data each revision was made from
    assert (
        store.read_manifest(tmp_path / "out" / "sync" / second)["metadata"][
            "raw_saved_at"
        ]
        == raw["saved_at"]
    )
    assert Flight.load(tmp_path / "out", sync_revision=False).sync_data is None
    with pytest.raises(ValueError):
        Flight.load(tmp_path / "out", sync_revision="rev_missing")


def test_migrate_legacy_files(tmp_path):
    import h5py

    from pils.migrate import migrate_proc
    from pils.utils.hdf5 import write_table

    proc = tmp_path / "proc"
    (proc / "azel").mkdir(parents=True)
    with h5py.File(proc / "azel" / "azel_solution.h5", "w") as f:
        group = f.create_group("SATP1").create_group("rev_20260101_000000")
        write_table(group, "azel_data", pl.DataFrame({"timestamp": [0.0], "az": [1.0]}))
        group.attrs["metadata"] = json.dumps({"telescope_name": "SATP1"})
    assert migrate_proc(proc, dry_run=True) == ["azel SATP1/rev_20260101_000000"]
    assert store.list_versions(proc / "azel", "SATP1") == []
    assert migrate_proc(proc) == ["azel SATP1/rev_20260101_000000"]
    tables, metadata = store.load_version(
        proc / "azel", "rev_20260101_000000", parent="SATP1"
    )
    assert tables["azel_data"].columns == ["timestamp", "az_deg"]
    assert metadata == {"telescope_name": "SATP1"}
    assert migrate_proc(proc) == []  # already converted
    assert (proc / "azel" / "azel_solution.h5").exists()  # kept
