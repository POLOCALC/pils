"""Tests for the multiplier output power / diode responsivity lab analysis."""

import h5py
import numpy as np
import polars as pl
import pytest

from pils.lab_analysis.multiplier_diode import (
    POWER_METER_OFFSET_DB,
    MultiplierDiodeAnalysis,
    read_raw,
    subtract_baseline,
    summarise_step,
)
from pils.lab_analysis.store import LabStore

NOTES = """---
config:
  components:
    multiplier:
      item_id: mul
    detector:
      item_id: det
  auxiliary_components:
    directional_coupler:
      item_id: cou
    waveguides:
      STR:
        item_id: str
---
"""
ITEMS = {
    "mul": {
        "item_id": "mul",
        "category": "multiplier",
        "name": "MUL_001",
        "specifications": {"output_band": "WR10"},
    },
    "det": {
        "item_id": "det",
        "category": "detector",
        "name": "DET_001",
        "specifications": {"input_band": "WR10"},
    },
    "cou": {
        "item_id": "cou",
        "category": "coupler",
        "name": "COU_001",
        "specifications": {"output_band": "WR10"},
    },
    "str": {
        "item_id": "str",
        "category": "straight",
        "name": "STR_001",
        "specifications": {"input_band": "WR10"},
    },
}
VNA_HEADER = "Frequency,S12 avg [dB],S12 std [dB],S21 avg [dB],S21 std [dB]\n"


def test_summarise_short_step():
    out = summarise_step(np.array([1.0, 3.0, -np.inf]), np.array([10.0, 20.0, 30.0]))
    assert out["power_mw"] == 2.0
    assert out["diode_output_mv"] == 20.0
    assert out["power_baseline_mw"] == 0.0


def test_summarise_on_off_step():
    off, on = np.full(30, 0.1), np.full(30, 0.6)
    power = np.concatenate([off, on] * 3)
    diode = np.concatenate([np.zeros(30), np.full(30, 5.0)] * 3)
    out = summarise_step(power, diode)
    assert out["power_mw"] == pytest.approx(0.5)
    assert out["power_err_mw"] == pytest.approx(0.0)
    assert out["diode_output_mv"] == pytest.approx(5.0)
    assert out["power_baseline_mw"] == pytest.approx(0.1)


def test_subtract_baseline():
    steps = pl.DataFrame(
        {
            "frequency_ghz": [93.0] * 3,
            "temperature_c": [0.0] * 3,
            "power_mw": [0.5, 1.0, 2.0],
            "power_baseline_mw": [0.1, 0.0, 0.0],
        }
    )
    out = subtract_baseline(steps)
    assert out["power_mw"].to_list() == pytest.approx([0.5, 0.9, 1.9])
    no_baseline = subtract_baseline(steps.with_columns(power_baseline_mw=pl.lit(0.0)))
    assert no_baseline["power_mw"].to_list() == [0.5, 1.0, 2.0]


@pytest.fixture
def lab(tmp_path):
    root = tmp_path / "lab" / "WR10"
    for item, test, rows in [
        ("coupler/COU_001", "transmission", "93,-0.7,0.01,-0.7,0.01\n"),
        ("coupler/COU_001", "coupling_performance", "93,-20.0,0.02,-20.0,0.02\n"),
        ("straight/STR_001", "transmission", "93,-0.3,0.01,-0.3,0.01\n"),
    ]:
        folder = root / item / test
        folder.mkdir(parents=True)
        (folder / "20260617_140000.csv").write_text(VNA_HEADER + rows)
    for item, test in [
        ("multiplier/MUL_001", "output_power"),
        ("detector/DET_001", "responsivity"),
    ]:
        folder = root / item / test
        folder.mkdir(parents=True)
        (folder / "20251101_140000.md").write_text(NOTES)
        with h5py.File(folder / "20251101_140000_raw.h5py", "w") as f:
            freq = f.create_group("93 GHz")
            freq.attrs["Frequency"] = 93
            temp = freq.create_group("0 C")
            temp.attrs["Temperature"] = 0.0
            for cv, power in [(0.7, 0.5), (0.8, 1.0)]:
                step = temp.create_group(f"{cv:.2f} V")
                step.attrs["control_voltage"] = cv
                step.attrs["PM range"] = 2.0
                step["Time [s]"] = np.arange(10.0)
                step["Power [mW]"] = np.full(10, power)
                step["Diode output [mV]"] = np.full(10, 100 * power)
    return tmp_path / "lab"


def test_run_corrects_and_saves_to_both_items(lab):
    store = LabStore(lab, get_item=ITEMS.__getitem__)
    folder = lab / "WR10" / "detector" / "DET_001" / "responsivity"
    analysis = MultiplierDiodeAnalysis.from_folder(folder)
    assert len(read_raw(analysis.measurement.raw_path)) == 2
    version = analysis.run_analysis(store=store, save_data=True)
    table = version["responsivity"]

    # Coupler output: add back the waveguide loss and the power-meter offset
    expected_dbm = 10 * np.log10(0.5) + 0.3 + POWER_METER_OFFSET_DB
    assert table["power_dbm"][0] == pytest.approx(expected_dbm)
    # Diode input: coupled fraction relative to the transmission
    expected_in = 10 ** (expected_dbm / 10) * 10 ** ((-20.0 + 0.7) / 10)
    assert table["diode_input_mw"][0] == pytest.approx(expected_in)
    assert table["diode_input_err_mw"][0] > 0

    # Saved in both the detector and the multiplier folders
    det = store.responsivity(ITEMS["det"])
    assert det.equals(table)
    mul_results = (
        lab
        / "WR10"
        / "multiplier"
        / "MUL_001"
        / "output_power"
        / "proc"
        / "multiplier_diode"
    )
    # <stem>/<rev>/ with the tables and a manifest (pils.store)
    saved = list(mul_results.glob("*/rev_*/manifest.json"))
    assert len(saved) == 1
    assert (saved[0].parent / "responsivity.parquet").is_file()


def test_missing_components(lab, tmp_path):
    folder = tmp_path / "other"
    folder.mkdir()
    (folder / "20250101_000000_raw.h5py").write_text("")
    (folder / "20250101_000000.md").write_text("---\nconfig:\n  components: {}\n---\n")
    analysis = MultiplierDiodeAnalysis.from_folder(folder)
    with pytest.raises(ValueError, match="do not name"):
        analysis.run_analysis(store=LabStore(lab, get_item=ITEMS.__getitem__))
