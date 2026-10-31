"""A telescope's wafers under a boresight rotation (pils.geometry.focal_plane)."""

import numpy as np
import pytest

from pils.geometry.focal_plane import (
    DEFAULT_WAFER_RADIUS_RAD,
    DEFAULT_WAFERS,
    Wafer,
    rotated_xieta,
    sky_of,
    wafer_polygon,
    wafer_quat,
)

pytest.importorskip("so3g")


def test_the_so_wafers_are_a_centre_and_six_around_it():
    centre, *ring = DEFAULT_WAFERS
    assert (centre.name, centre.xi, centre.eta) == ("ws0", 0.0, 0.0)
    assert np.allclose([np.hypot(w.xi, w.eta) for w in ring], 0.221, atol=0.001)


def test_a_rotation_turns_a_point_around_the_centre():
    x, e = rotated_xieta(0.1, 0.0, 90.0)
    assert np.hypot(x, e) == pytest.approx(0.1, abs=1e-6)
    assert rotated_xieta(0.1, 0.05, 0.0) == (pytest.approx(0.1), pytest.approx(0.05))


def test_a_wafer_is_a_hexagon_unless_it_has_an_outline():
    hexagon = wafer_polygon(DEFAULT_WAFERS[0], 0.0)
    assert hexagon.shape == (6, 2)
    assert np.allclose(
        np.hypot(hexagon[:, 0], hexagon[:, 1]), DEFAULT_WAFER_RADIUS_RAD, atol=1e-4
    )
    own = Wafer("w1", 0.0, 0.0, outline=((0.01, 0.0), (0.0, 0.01), (-0.01, 0.0)))
    assert wafer_polygon(own, 0.0).shape == (3, 2)


def test_the_boresight_s_own_point_is_where_it_points():
    az, el = sky_of(185.0, 45.0, 0.0, 0.0)
    assert (az, el) == (pytest.approx(185.0), pytest.approx(45.0))
    assert wafer_quat(DEFAULT_WAFERS[0], 30.0) is not None


def test_the_same_as_gose_s_planner():
    gose = pytest.importorskip("gose.sat_pointing")
    for w in DEFAULT_WAFERS:
        assert np.array_equal(wafer_polygon(w, -30.0), gose.wafer_polygon(w, -30.0))
        assert rotated_xieta(w.xi, w.eta, 30.0) == gose.rotated_xieta(w.xi, w.eta, 30.0)
    assert sky_of(185.0, 45.0, 0.1, -0.05) == gose.sky_of(185.0, 45.0, 0.1, -0.05)

    def values(wafers):
        return [(w.name, w.xi, w.eta, w.outline) for w in wafers]

    assert values(DEFAULT_WAFERS) == values(gose.DEFAULT_WAFERS)
