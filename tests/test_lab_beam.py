"""Tests for the antenna beam lab analysis."""

import h5py
import numpy as np
import polars as pl
import pytest

from pils.lab_analysis.beam import (
    FWHM_PER_SIGMA,
    BeamAnalysis,
    expected_fwhm_deg,
    gain_from_fwhm,
)
from pils.lab_analysis.store import LabStore

NOTES = """---
config:
  environment: "anechoic chamber"
  components:
    receiving_antenna:
      item_id: ant-1
    launching_antenna:
      item_id: ant-7
  auxiliary_components:
    grid_status:
      False
---
"""
FWHM = {"Hplane": 18.0, "Eplane": 14.0, "Dplane": 16.0}


@pytest.fixture
def beam_folder(tmp_path):
    folder = tmp_path / "lab" / "WR10" / "antenna" / "ANT_001" / "radiation_pattern"
    folder.mkdir(parents=True)
    angles = np.arange(-90, 90.5, 0.5)
    freqs = [84.0, 93.0]
    with h5py.File(folder / "20251021_140000_raw.h5py", "w") as f:
        for plane, fwhm in FWHM.items():
            for pol, level in [("copol", 0.0), ("crosspol", -30.0)]:
                ang = np.tile(angles, len(freqs))
                freq = np.repeat(freqs, len(angles))
                sigma = fwhm / FWHM_PER_SIGMA * (93.0 / freq)  # narrower at 93 GHz
                linear = np.exp(-(ang**2) / sigma**2) + 1e-6
                g = f.create_group(f"{plane}/{pol}")
                g["Ang(DEG)"] = ang
                g["Freq(GHz)"] = freq
                g["S12(DB)"] = 10 * np.log10(linear) - 40 + level
    (folder / "20251021_140000.md").write_text(NOTES)
    return folder


def test_expected_fwhm():
    # 4π / (1.133 θ²) = G
    theta = np.radians(expected_fwhm_deg(15.0))
    assert 4 * np.pi / (1.133 * theta**2) == pytest.approx(10**1.5)


def test_fit_and_gain(beam_folder, tmp_path):
    beam = BeamAnalysis.from_folder(beam_folder, output_dir=tmp_path / "out")
    version = beam.run_analysis(nominal_gain_dbi=15.0, save_data=True, plot=True)
    fwhm = version["fwhm"]
    at93 = fwhm.filter(pl.col("frequency_ghz") == 93.0)
    for plane, expected in FWHM.items():
        row = at93.filter(pl.col("plane") == plane)
        assert row["fwhm_deg"][0] == pytest.approx(expected, rel=1e-3)
        assert row["centre_deg"][0] == pytest.approx(0.0, abs=1e-3)
    gain = version["gain"].filter(pl.col("frequency_ghz") == 93.0)
    omega = 1.13 * np.radians(18.0) * np.radians(14.0)
    assert gain["gain_dbi"][0] == pytest.approx(
        10 * np.log10(4 * np.pi / omega), abs=1e-3
    )
    assert gain["method"][0] == "H+E planes"
    assert version.metadata["grid_status"] is False
    assert beam.get_latest_version()["fwhm"].height == 6
    figures = beam.figures_dir(version.version_name)
    assert (figures / "beam_cuts.png").is_file()

    # The store reads the PILS run
    item = {
        "category": "antenna",
        "name": "ANT_001",
        "specifications": {"band": "WR10"},
    }
    store = LabStore(tmp_path / "out", get_item=None)
    store.item_folder = lambda _: tmp_path / "out"  # results live under output_dir
    store.test_folder = lambda _item, _test: tmp_path / "out"
    assert store.antenna_gain(item).height == 2


def test_nominal_gain_from_stout_item(beam_folder):
    store = LabStore(get_item=lambda item_id: {"specifications": {"gain": 15.0}})
    beam = BeamAnalysis.from_folder(beam_folder)
    assert beam.nominal_gain(store) == 15.0


def test_gain_errors():
    fwhm = pl.DataFrame(
        {
            "plane": ["Hplane", "Eplane"],
            "frequency_ghz": [93.0, 93.0],
            "fwhm_deg": [18.0, 14.0],
            "fwhm_err_deg": [0.18, 0.0],
        }
    )
    gain = gain_from_fwhm(fwhm)
    # 1 % on θ_H -> 1 % on Ω -> 4.343 * 0.01 dB
    assert gain["gain_err_db"][0] == pytest.approx(10 / np.log(10) * 0.01)

    d_only = pl.DataFrame(
        {
            "plane": ["Dplane"],
            "frequency_ghz": [93.0],
            "fwhm_deg": [16.0],
            "fwhm_err_deg": [0.16],
        }
    )
    gain = gain_from_fwhm(d_only)
    assert gain["method"][0] == "D plane"
    assert gain["gain_err_db"][0] == pytest.approx(10 / np.log(10) * 0.02)
    with pytest.raises(ValueError):
        gain_from_fwhm(d_only.with_columns(pl.lit("Xplane").alias("plane")))
