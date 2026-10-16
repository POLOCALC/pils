"""Tests for lab measurement notes and the lab results store."""

import sys
import types
from unittest.mock import patch

import h5py
import numpy as np
import polars as pl
import pytest

from pils.lab_analysis.notes import (
    LabMeasurement,
    component_ids,
    list_measurements,
    load_notes,
)
from pils.lab_analysis.store import LabDataNotFound, LabStore, _stout_item, value_at
from pils.utils.hdf5 import save_version

NOTES = """---
config:
  environment: "climate chamber"
  components:
    multiplier:
      item_id: mul-1
    detector:
      item_id: det-1
  auxiliary_components:
    directional_coupler:
      item_id: cou-1
    waveguides:
      STR:
        item_id: str-1
    grid_status: False
---

# Measurement description
"""

COUPLER = {
    "category": "coupler",
    "name": "COU_001",
    "specifications": {"input_band": "WR10", "output_band": "WR10"},
}
ANTENNA = {"category": "antenna", "name": "ANT_001", "specifications": {"band": "WR10"}}
DETECTOR = {
    "category": "detector",
    "name": "DET_001",
    "specifications": {"input_band": "WR10"},
}


@pytest.fixture
def lab(tmp_path):
    root = tmp_path / "lab"
    folder = root / "WR10" / "coupler" / "COU_001" / "transmission"
    folder.mkdir(parents=True)
    (folder / "20260617_140000_raw.s2p").write_text("")
    (folder / "20260617_140000.csv").write_text(
        "Frequency,S12 avg [dB],S12 std [dB],S21 avg [dB],S21 std [dB]\n"
        "84,-0.70,0.01,-0.80,0.02\n93,-0.72,0.01,-0.82,0.02\n"
    )
    return root


class TestNotes:
    def test_load_notes_and_ids(self, tmp_path):
        path = tmp_path / "20251101_140000.md"
        path.write_text(NOTES)
        config = load_notes(path)
        assert config["environment"] == "climate chamber"
        assert component_ids(config) == {
            "multiplier": "mul-1",
            "detector": "det-1",
            "aux.directional_coupler": "cou-1",
            "aux.waveguides.STR": "str-1",
        }

    def test_notes_without_front_matter(self, tmp_path):
        path = tmp_path / "a.md"
        path.write_text("# Just text\n")
        with pytest.raises(ValueError, match="No configuration block"):
            load_notes(path)

    def test_find_measurement(self, tmp_path):
        for stem in ["20250101_100000", "20250202_100000"]:
            (tmp_path / f"{stem}_raw.h5py").write_text("")
        (tmp_path / "20250202_100000.md").write_text(NOTES)
        (tmp_path / "20250202_100000_processed.h5py").write_text("")
        assert list_measurements(tmp_path) == ["20250101_100000", "20250202_100000"]
        latest = LabMeasurement.find(tmp_path)
        assert latest.stem == "20250202_100000"
        assert latest.raw_path.name == "20250202_100000_raw.h5py"
        assert latest.config["components"]["detector"]["item_id"] == "det-1"
        older = LabMeasurement.find(tmp_path, "20250101_100000")
        assert older.notes_path is None and older.config == {}
        with pytest.raises(FileNotFoundError):
            LabMeasurement.find(tmp_path, "20990101_000000")
        with pytest.raises(FileNotFoundError):
            LabMeasurement.find(tmp_path / "missing")


class TestStore:
    def test_item_folder_picks_existing_band(self, lab):
        store = LabStore(lab, get_item=lambda _: COUPLER)
        item = {
            "category": "coupler",
            "name": "COU_001",
            "specifications": {"input_band": "WR06", "output_band": "WR10"},
        }
        assert store.item_folder(item) == lab / "WR10" / "coupler" / "COU_001"
        with pytest.raises(LabDataNotFound):
            store.item_folder({"category": "x", "name": "y", "specifications": {}})
        assert store.item("cou-1") is COUPLER

    def test_legacy_vna_csv_is_relabelled(self, lab):
        store = LabStore(lab)
        # Legacy "S12" is the Touchstone S21 column
        assert store.transmission(COUPLER, "transmission", 93) == (-0.72, 0.01)
        summary = store.vna_summary(COUPLER, "transmission")
        assert summary["s12_avg_db"].to_list() == [-0.80, -0.82]
        with pytest.raises(LabDataNotFound, match="available"):
            store.transmission(COUPLER, "transmission", 90)

    def test_saved_run_takes_precedence(self, lab):
        from pils import store

        folder = lab / "WR10" / "coupler" / "COU_001" / "transmission"
        summary = pl.DataFrame(
            {"frequency_ghz": [93.0], "s21_avg_db": [-0.4], "s21_std_db": [0.02]}
        )
        store.save_version(
            folder / "proc" / "vna",
            "rev_20260101_120000",
            {"summary": summary},
            {},
            parent="20260617_140000",
        )
        assert LabStore(lab).transmission(COUPLER, "transmission", 93) == (-0.4, 0.02)

    def test_legacy_hdf5_run_takes_precedence(self, lab):
        folder = lab / "WR10" / "coupler" / "COU_001" / "transmission"
        summary = pl.DataFrame(
            {"frequency_ghz": [93.0], "s21_avg_db": [-0.5], "s21_std_db": [0.03]}
        )
        save_version(
            folder / "proc" / "vna" / "vna_solution.h5",
            "rev_20260101_120000",
            {"summary": summary},
            {},
            parent="20260617_140000",
        )
        assert LabStore(lab).transmission(COUPLER, "transmission", 93) == (-0.5, 0.03)

    def test_legacy_antenna_files(self, lab):
        folder = lab / "WR10" / "antenna" / "ANT_001"
        (folder / "insertion_loss").mkdir(parents=True)
        (folder / "insertion_loss" / "20260609_140000.csv").write_text(
            "Frequency [GHz],Insertion Loss [dB],Err Insertion Loss [dB]\n93,-0.07,0.01\n"
        )
        (folder / "impedance_matching").mkdir()
        with h5py.File(
            folder / "impedance_matching" / "20260617_140000.h5py", "w"
        ) as f:
            f["fsp_impedance_kapton/frequency [GHz]"] = [93.0]
            f["fsp_impedance_kapton/S11 avg [dB]"] = [-18.0]
            f["fsp_impedance_kapton/S11 std [dB]"] = [0.02]
        (folder / "radiation_pattern").mkdir()
        (folder / "radiation_pattern" / "20251021_gain_processed.csv").write_text(
            "Frequency [GHz],Gain [dBi],err gain [dB]\n93.0,20.1,0.01\n"
        )
        with h5py.File(
            folder / "radiation_pattern" / "20251021_fwhm_processed.hdf5", "w"
        ) as f:
            for plane in ["Hplane", "Eplane"]:
                f[f"{plane}/Frequency [GHz]"] = [93.0]
                f[f"{plane}/FWHM [deg]"] = [16.0]
                f[f"{plane}/err FWHM [deg]"] = [0.1]

        store = LabStore(lab)
        assert store.insertion_loss(ANTENNA, 93) == (-0.07, 0.01)
        assert store.reflection(ANTENNA, 93) == (-18.0, 0.02)
        with pytest.raises(LabDataNotFound, match="fsp_impedance_kapton"):
            store.reflection(ANTENNA, 93, dataset="other")
        assert store.antenna_gain(ANTENNA)["gain_dbi"].to_list() == [20.1]
        fwhm = store.antenna_fwhm(ANTENNA)
        assert set(fwhm["plane"]) == {"Hplane", "Eplane"}

    def test_legacy_responsivity(self, lab):
        folder = lab / "WR10" / "detector" / "DET_001" / "responsivity"
        folder.mkdir(parents=True)
        with h5py.File(folder / "20251101_140000_processed.h5py", "w") as f:
            for temp in ["-10 C", "10 C"]:
                g = f.create_group(f"93 GHz/{temp}")
                g["Control Voltage [V]"] = [0.7, 0.8]
                g["Diode output [mV]"] = [1.0, 2.0]
                g["err diode output [mV]"] = [0.1, 0.1]
                g["Diode input pwr [mW]"] = [0.01, 0.02]
                g["err diode input [mW]"] = [0.001, 0.001]
        table = LabStore(lab).responsivity(DETECTOR)
        assert table.shape[0] == 4
        assert table["temperature_c"].unique().sort().to_list() == [-10.0, 10.0]
        assert np.allclose(table["control_voltage_v"].to_list()[:2], [0.7, 0.8])

    def test_missing_results(self, lab):
        with pytest.raises(LabDataNotFound):
            LabStore(lab).antenna_gain(ANTENNA)

    def test_value_at(self):
        table = pl.DataFrame({"frequency_ghz": [84.0, 93.0], "v": [1, 2], "e": [3, 4]})
        assert value_at(table, 93, "v", "e") == (2.0, 4.0)


class TestStoutItem:
    """Items looked up by id in the STOUT inventory."""

    @pytest.fixture
    def inventory(self):
        class NotFoundError(Exception):
            pass

        items = {"det-1": {"item_id": "det-1", "name": "detector"}}

        class InventoryService:
            def get_item_by_id(self, item_id):
                if item_id not in items:
                    raise NotFoundError(item_id)
                return items[item_id]

        modules = {
            "stout": types.ModuleType("stout"),
            "stout.exceptions": types.SimpleNamespace(NotFoundError=NotFoundError),
            "stout.services": types.ModuleType("stout.services"),
            "stout.services.inventory": types.ModuleType("stout.services.inventory"),
            "stout.services.inventory.service": types.SimpleNamespace(
                InventoryService=InventoryService
            ),
        }
        with patch.dict(sys.modules, modules):
            yield

    def test_found(self, inventory):
        assert _stout_item("det-1")["name"] == "detector"

    def test_missing_item_is_lab_data_not_found(self, inventory):
        with pytest.raises(LabDataNotFound, match="missing"):
            _stout_item("missing")

    def test_without_stout(self):
        with patch.dict(sys.modules, {"stout": None}):
            with pytest.raises(ImportError, match="pils\\[stout\\]"):
                _stout_item("det-1")


def test_legacy_and_saved_runs_listed_and_migrated(tmp_path):
    """Runs in a legacy <analysis>_solution.h5 are listed, loaded and migrated."""
    from pils import store
    from pils.lab_analysis.base import LabAnalysis
    from pils.migrate import migrate_lab

    class Fake(LabAnalysis):
        analysis_name = "vna"

    class Measurement:
        folder = tmp_path
        stem = "20260617_140000"

    analysis = Fake(Measurement())  # type: ignore[arg-type]
    table = pl.DataFrame({"frequency_ghz": [93.0], "s21_avg_db": [-0.5]})
    save_version(
        analysis.results_path,
        "rev_20260101_000000",
        {"summary": table},
        {"a": 1},
        parent="20260617_140000",
    )
    store.save_version(
        analysis.results_dir,
        "rev_20260202_000000",
        {"summary": table},
        {"b": 2},
        parent="20260617_140000",
    )
    assert analysis.list_versions() == ["rev_20260101_000000", "rev_20260202_000000"]
    assert analysis.get_version("rev_20260101_000000").metadata == {"a": 1}
    assert analysis.get_latest_version().metadata == {"b": 2}

    proc = tmp_path / "proc"
    assert migrate_lab(proc, dry_run=True) == [
        "vna 20260617_140000/rev_20260101_000000"
    ]
    assert migrate_lab(proc) == ["vna 20260617_140000/rev_20260101_000000"]
    assert migrate_lab(proc) == []
    tables, metadata = store.load_version(
        proc / "vna", "rev_20260101_000000", parent="20260617_140000"
    )
    assert tables["summary"].equals(table) and metadata == {"a": 1}
    assert analysis.results_path.is_file()  # kept
