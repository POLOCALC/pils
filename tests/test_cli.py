"""Tests for the pils command line."""

import numpy as np
import pytest

from pils.cli import build_parser, main


def test_parser():
    args = build_parser().parse_args(["power", "flight_x", "-t", "satp1", "-w", "ws0"])
    assert (args.telescope, args.wafer_slot, args.bandpass) == ("satp1", "ws0", "gab")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["lab", "unknown", "folder"])


def test_lab_vna_dry_run_and_save(tmp_path, capsys):
    folder = tmp_path / "attenuation"
    folder.mkdir()
    lines = ["# GHz S dB R 50\n"]
    for f in np.arange(83.0, 103.0, 0.05):
        lines.append(f"{f} -20 0 -3 0 -3 0 -20 0\n")
    (folder / "20260617_140000_raw.s2p").write_text("".join(lines))

    assert main(["lab", "vna", str(folder), "--dry-run"]) == 0
    assert "not saved" in capsys.readouterr().out
    assert not (folder / "proc").exists()

    out = tmp_path / "out"
    assert main(["lab", "vna", str(folder), "--output-dir", str(out), "--no-plot"]) == 0
    assert (out / "proc" / "vna" / "vna_solution.h5").is_file()
