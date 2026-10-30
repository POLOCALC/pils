"""A telescope's detectors: positions, bands and beams (pils.geometry.detectors)."""

import numpy as np
import pytest

from pils.geometry.detectors import (
    band_of_source,
    focal_plane_model,
    nominal_focal_plane,
    pixel_positions,
    rotated_eta,
    rotated_pixels,
    sees,
    source_label,
)
from pils.geometry.focal_plane import DEFAULT_WAFERS, Wafer


def test_a_source_names_its_band():
    assert band_of_source(90) == band_of_source("90") == "SAT_f090"
    assert (
        band_of_source(220) == "SAT_f230" and band_of_source("SAT_f150") == "SAT_f150"
    )
    assert source_label("SAT_f090") == "90 GHz"
    assert sees("SAT_MF", 150) and not sees("SAT_MF", 220)
    with pytest.raises(ValueError):
        band_of_source(100)


def test_the_so_layouts_are_shipped():
    mf = nominal_focal_plane("SAT_MF")
    assert mf.n_pixels == 430 and mf.bands == ("SAT_f090", "SAT_f150")
    assert mf.fwhm_arcmin == {"SAT_f090": 30.0, "SAT_f150": 17.0}
    # centred on the wafer
    assert abs(mf.xi.mean()) < 1e-9 and abs(mf.eta.mean()) < 1e-9
    assert nominal_focal_plane("SAT_LF").n_pixels == 37


def test_the_pixels_are_placed_on_each_wafer():
    pos = pixel_positions(DEFAULT_WAFERS, "SAT_MF")
    assert set(pos) == {w.name for w in DEFAULT_WAFERS}
    for w in DEFAULT_WAFERS:
        assert pos[w.name].shape == (430, 2)
        assert np.allclose(pos[w.name].mean(axis=0), [w.xi, w.eta], atol=1e-9)
    # a wafer of a detector table keeps its own pixels
    own = Wafer("w1", 0.0, 0.0, pixels=np.array([[0.01, 0.0], [0.0, 0.01]]))
    assert np.array_equal(pixel_positions([own])["w1"], own.pixels)


def test_a_telescope_with_its_own_table_has_its_own_beam():
    class Tel:
        name, wafer_type, beam_fwhm_arcmin = "CLASS_w1", "CLASS_W", 20.0
        wafers = (Wafer("w1", 0.0, 0.0),)

    model = focal_plane_model(Tel())
    assert model.n_pixels == 0 and model.fwhm_arcmin == {"SAT_f090": 20.0}


def test_rotated_pixels_are_in_degrees_and_turned():
    straight = rotated_pixels(DEFAULT_WAFERS[:1], "SAT_MF", 0.0)["ws0"]
    turned = rotated_pixels(DEFAULT_WAFERS[:1], "SAT_MF", 30.0)["ws0"]
    radius = lambda p: np.hypot(p[:, 0], p[:, 1])  # noqa: E731
    assert np.allclose(radius(straight), radius(turned), atol=1e-6)
    assert not np.allclose(straight, turned)
    assert np.abs(straight).max() < 10  # degrees, not radians
    xy = pixel_positions(DEFAULT_WAFERS[:1])["ws0"]
    assert np.allclose(rotated_eta(xy[:, 0], xy[:, 1], 0.0), xy[:, 1], atol=1e-9)


def test_the_same_as_gose_s_planner():
    detectors = pytest.importorskip("gose.detectors")
    crossings = pytest.importorskip("gose.crossings")
    for wafer_type in ("SAT_LF", "SAT_MF", "SAT_UHF"):
        a, b = (
            nominal_focal_plane(wafer_type),
            detectors.nominal_focal_plane(wafer_type),
        )
        assert np.array_equal(a.xi, b.xi) and np.array_equal(a.eta, b.eta)
        assert a.bands == b.bands and dict(a.fwhm_arcmin) == dict(b.fwhm_arcmin)
        for bs in (-30.0, 0.0, 30.0):
            mine = rotated_pixels(DEFAULT_WAFERS, wafer_type, bs)
            theirs = crossings.rotated_pixels(DEFAULT_WAFERS, wafer_type, bs)
            assert all(np.array_equal(mine[w], theirs[w]) for w in mine)
