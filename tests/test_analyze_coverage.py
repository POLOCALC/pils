"""The focal-plane coverage of a flown flight (pils.analyze.coverage)."""

import numpy as np
import polars as pl
import pytest

from pils import store
from pils.analyze.coverage import flight_coverage, thin
from pils.geometry.detectors import nominal_focal_plane
from pils.geometry.focal_plane import DEFAULT_WAFERS

pytest.importorskip("so3g")


def _flight(tmp_path, *, boresight=0.0, offset=0.0):
    """A flight in which the drone crosses SATP3's boresight along the azimuth, in 60 s."""
    t = 1_700_000_000 + np.arange(0, 60, 0.05)
    store.save_version(
        tmp_path / "proc" / "azel" / "SATP3",
        "rev_20260101_000000",
        {"azel_data": pl.DataFrame({"timestamp": t, "az_deg": np.linspace(178, 182, t.size),
                                    "el_deg": np.full(t.size, 50.0)})},
        {},
    )
    store.save_version(
        tmp_path / "proc" / "telescope" / "SATP3",
        "rev_20260101_000000",
        {"pointing": pl.DataFrame({"timestamp": t, "az_enc_deg": np.full(t.size, 180.0 + offset),
                                   "el_enc_deg": np.full(t.size, 50.0),
                                   "boresight_enc_deg": np.full(t.size, boresight),
                                   "flagged": np.zeros(t.size, bool)})},
        {},
    )
    return tmp_path


def _coverage(flight, source=90, **kw):
    return flight_coverage(flight, "satp3", source, wafers=DEFAULT_WAFERS,
                           focal_plane=nominal_focal_plane("SAT_MF"), **kw)


def test_the_drone_crosses_the_detectors_on_its_path(tmp_path):
    out = _coverage(_flight(tmp_path))
    assert out["telescope"] == "SATP3" and out["band"] == "SAT_f090"
    assert out["beam_fwhm_arcmin"] == 30.0 and out["radius_deg"] == pytest.approx(0.25)
    px = out["pixels"]
    assert len(px["wafer"]) == len(px["xi"]) == len(px["crossings"]) == 7 * 430
    crossed = np.array(px["crossings"]) > 0
    xi, eta = np.abs(np.array(px["xi"])), np.abs(np.array(px["eta"]))
    # the path is about eta = 0 for |xi| < 4 cos(50 deg) / 2 = 1.29 deg: a detector is crossed when it
    # passes within 0.25 deg of it
    assert out["crossed"] == crossed.sum() > 0
    assert crossed[(eta < 0.2) & (xi < 1.2)].all()
    assert not crossed[(eta > 0.3) | (xi > 1.6)].any()
    assert out["datasets"] == {"azel": "rev_20260101_000000", "pointing": "rev_20260101_000000"}


def test_the_window_and_the_encoder_offset(tmp_path):
    flight = _flight(tmp_path, offset=1.0)
    whole = _coverage(flight, az_offset_deg=1.0)
    # the encoder's azimuth less the offset is the true one
    assert whole["pixels"] == _coverage(_flight(tmp_path / "plain"))["pixels"]
    part = _coverage(flight, az_offset_deg=1.0, start=1_700_000_000.0, end=1_700_000_010.0)
    assert part["window"][1] <= 1_700_000_010 and part["samples"] < whole["samples"]
    # the first 10 s are at least 1.3 deg from the boresight: only the edge of the plane
    assert part["crossed"] < whole["crossed"]


def test_what_is_missing_is_said(tmp_path):
    with pytest.raises(ValueError, match="no azel/SATP3"):
        _coverage(tmp_path)
    flight = _flight(tmp_path)
    with pytest.raises(ValueError, match="no band of SATP3 for 220 GHz"):
        _coverage(flight, source=220)
    with pytest.raises(ValueError, match="no pointing while the drone was seen"):
        _coverage(flight, start=0.0, end=1.0)


def test_the_path_is_thinned():
    xi, eta = thin(np.arange(10_000.0), np.zeros(10_000), max_points=100)
    assert len(xi) == len(eta) == 100 and xi[0] == 0 and xi[-1] == 9999
