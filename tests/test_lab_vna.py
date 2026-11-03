"""Tests for the VNA lab analysis."""

import numpy as np
import pytest

from pils.lab_analysis.store import LabStore
from pils.lab_analysis.vna import (
    VNAAnalysis,
    frequency_summary,
    mismatch_corrected,
    read_touchstone,
)

HEADER = (
    "!Keysight Technologies,N5245B\n"
    "!S2P File: Measurements: S11, S21, S12, S22:\n"
    "# Hz S  dB   R 50\n"
)


def write_s2p(path, freqs_ghz, s11=-20.0, s21=-3.0, s12=-3.5, s22=-15.0):
    lines = [HEADER]
    for i, f in enumerate(freqs_ghz):
        jitter = 0.01 * (-1) ** i
        lines.append(f"{f * 1e9:.0f} {s11} 10 {s21 + jitter} 20 {s12} 30 {s22} 40\n")
    path.write_text("".join(lines))


@pytest.fixture
def measurement_folder(tmp_path):
    folder = tmp_path / "lab" / "WR10" / "attenuator" / "ATT_009" / "attenuation"
    folder.mkdir(parents=True)
    write_s2p(folder / "20260617_140000_raw.s2p", np.arange(83.0, 103.0, 0.05))
    return folder


def test_read_touchstone_column_order(measurement_folder):
    spectrum, header = read_touchstone(measurement_folder / "20260617_140000_raw.s2p")
    assert header["format"] == "db" and header["frequency_unit"] == "hz"
    assert spectrum["frequency_ghz"][0] == pytest.approx(83.0)
    # Touchstone order: S11, S21, S12, S22
    assert spectrum["s12_db"][0] == -3.5
    assert spectrum["s21_deg"][0] == 20
    assert spectrum["s22_db"][0] == -15.0


def test_read_touchstone_magnitude_angle(tmp_path):
    path = tmp_path / "ma_raw.s2p"
    path.write_text("# GHz S MA R 50\n90 0.1 0 0.5 0 0.5 0 0.1 0\n")
    spectrum, _ = read_touchstone(path)
    assert spectrum["s21_db"][0] == pytest.approx(20 * np.log10(0.5))
    assert spectrum["frequency_ghz"][0] == 90


def test_mismatch_correction():
    import polars as pl

    spectrum = pl.DataFrame(
        {
            "frequency_ghz": [90.0],
            "s11_db": [-3.0103],  # half the power reflected
            "s21_db": [-6.0206],
            "s12_db": [-6.0206],
            "s22_db": [-100.0],
        }
    )
    corrected = mismatch_corrected(spectrum)
    assert corrected["s21_corrected_db"][0] == pytest.approx(-3.0103, abs=1e-3)
    assert corrected["s12_corrected_db"][0] == pytest.approx(-6.0206, abs=1e-3)


def test_frequency_summary_window(measurement_folder):
    spectrum, _ = read_touchstone(measurement_folder / "20260617_140000_raw.s2p")
    summary = frequency_summary(spectrum, [90.0], window_ghz=0.1)
    assert summary["s21_avg_db"][0] == pytest.approx(-3.0, abs=0.01)
    assert summary["s21_std_db"][0] == pytest.approx(0.01, abs=1e-3)
    with pytest.raises(ValueError, match="No VNA points"):
        frequency_summary(spectrum, [200.0])


def test_run_save_and_store(measurement_folder, tmp_path):
    out = tmp_path / "results"
    vna = VNAAnalysis.from_folder(measurement_folder, output_dir=out)
    assert not (out / "proc").exists()  # nothing written before a save
    version = vna.run_analysis(save_data=True, plot=True)
    assert version["summary"].height == 7
    assert vna.list_versions() == [version.version_name]
    loaded = vna.get_latest_version()
    assert loaded.metadata["measurement"] == "20260617_140000"
    assert loaded["summary"].equals(version["summary"])
    figures = vna.figures_dir(version.version_name)
    assert (figures / "s_parameters.png").is_file()
    assert (figures / "s21_frequency_zoom.png").is_file()


def test_store_reads_pils_run(measurement_folder):
    VNAAnalysis.from_folder(measurement_folder).run_analysis(save_data=True)
    item = {
        "category": "attenuator",
        "name": "ATT_009",
        "specifications": {"band": "WR10"},
    }
    store = LabStore(measurement_folder.parents[3])
    value, err = store.transmission(item, "attenuation", 93.0)
    assert value == pytest.approx(-3.0, abs=0.01)
    assert err > 0
