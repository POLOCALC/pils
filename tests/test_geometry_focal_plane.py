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


def test_an_offset_point_is_off_the_boresight_by_its_distance():
    xi, eta = 0.02, -0.01  # rad
    az, el = sky_of(185.0, 45.0, xi, eta)
    assert el < 45.0 and az > 185.0  # +xi to larger azimuth, -eta lower
    a = np.radians([185.0, 45.0, az, el])
    cos_d = np.sin(a[1]) * np.sin(a[3]) + np.cos(a[1]) * np.cos(a[3]) * np.cos(
        a[2] - a[0]
    )
    assert np.arccos(cos_d) == pytest.approx(np.hypot(xi, eta), rel=1e-3)
