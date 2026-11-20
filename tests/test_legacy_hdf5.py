"""The read-only reader of the HDF5 results PILS saved before the parquet store."""

import h5py
import numpy as np
import polars as pl
import pytest

from pils import legacy_hdf5

from .legacy_hdf5_writer import save_version


def test_save_and_load(tmp_path):
    path = tmp_path / "proc" / "x" / "x_solution.h5"
    frame = pl.DataFrame({"freq": [84.0, 87.0], "count": [1, 2], "plane": ["H", "E"]})
    save_version(path, "rev_20260101_120000", {"gain": frame}, {"n": np.int64(3)})
    tables, metadata = legacy_hdf5.load_version(path, "rev_20260101_120000")
    assert tables["gain"].equals(frame)
    assert metadata == {"n": 3}


def test_nulls_and_types_come_back(tmp_path):
    path = tmp_path / "a.h5"
    frame = pl.DataFrame(
        {
            "x": [1.0, None, 3.0],
            "n": [1, 2, None],
            "s": ["a", None, "c"],
            "t": pl.Series([1, 2, None], dtype=pl.Datetime("us", "UTC")),
            "ok": [True, None, False],
        }
    )
    save_version(path, "rev_20260101_120000", {"t": frame}, {})
    tables, _ = legacy_hdf5.load_version(path, "rev_20260101_120000")
    assert tables["t"].equals(frame)
    assert tables["t"].schema == frame.schema


def test_versions_with_parent(tmp_path):
    path = tmp_path / "a.h5"
    frame = pl.DataFrame({"x": [1.0]})
    save_version(path, "rev_20260101_120000", {"t": frame}, {}, parent="SATP1")
    save_version(path, "rev_20260102_120000", {"t": frame}, {}, parent="SATP1")
    save_version(path, "rev_20260103_120000", {"t": frame}, {}, parent="SATP3")
    assert legacy_hdf5.list_versions(path, parent="SATP1") == [
        "rev_20260101_120000",
        "rev_20260102_120000",
    ]
    assert legacy_hdf5.list_versions(path, parent="NOPE") == []
    assert legacy_hdf5.list_versions(tmp_path / "missing.h5") == []
    with h5py.File(path) as f:
        assert f.attrs["latest_version"] == "rev_20260103_120000"


def test_missing_version_and_file(tmp_path):
    path = tmp_path / "a.h5"
    save_version(path, "rev_20260101_120000", {}, {})
    with pytest.raises(KeyError):
        legacy_hdf5.load_version(path, "rev_20990101_000000")
    with pytest.raises(FileNotFoundError):
        legacy_hdf5.load_version(tmp_path / "missing.h5", "rev_20260101_120000")
