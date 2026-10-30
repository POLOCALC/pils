"""The drone in a telescope's focal plane, and the detectors it crosses (pils.geometry.crossings)."""

import numpy as np
import pytest

from pils.geometry.crossings import crossing_times, crossings, relative_path, rotate

pytest.importorskip("so3g")


def test_the_drone_at_the_boresight_is_at_the_centre():
    xi, eta = relative_path(
        np.array([180.0]), np.array([50.0]), np.array([180.0]), np.array([50.0])
    )
    assert np.allclose([xi[0], eta[0]], [0, 0], atol=1e-9)


def test_a_drone_above_the_boresight_is_up_in_the_focal_plane():
    xi, eta = relative_path(
        np.array([180.0]), np.array([50.0]), np.array([180.0]), np.array([51.0])
    )
    # projected coordinates: one degree up is 0.99995
    assert xi[0] == pytest.approx(0, abs=1e-9) and eta[0] == pytest.approx(
        1.0, abs=1e-3
    )
    # beside it: along xi, by the azimuth times cos(elevation)
    xi, eta = relative_path(
        np.array([180.0]), np.array([50.0]), np.array([181.0]), np.array([50.0])
    )
    assert (
        abs(xi[0]) == pytest.approx(np.cos(np.radians(50)), abs=1e-3)
        and abs(eta[0]) < 0.01
    )


def test_a_rotation_turns_the_focal_plane_and_keeps_distances():
    pts = rotate(np.array([1.0, 0.0]), np.array([0.0, 2.0]), 30.0)
    assert np.allclose(np.hypot(pts[:, 0], pts[:, 1]), [1.0, 2.0], atol=1e-6)
    assert np.allclose(
        rotate(np.array([1.0]), np.array([0.5]), 0.0), [[1.0, 0.5]], atol=1e-9
    )


def test_crossings_count_each_entry_into_a_beam():
    pixels = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 0.3]])
    # back and forth along xi through the first pixel: in, out, in again
    xi = np.concatenate([np.linspace(-1, 1, 201), np.linspace(1, -1, 201)])
    eta = np.zeros_like(xi)
    n = crossings(xi, eta, pixels, radius_deg=0.2)
    assert n.tolist() == [2, 0, 0]
    # starting inside counts once
    assert crossings(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]), pixels, 0.2
    ).tolist() == [1, 0, 0]


def test_crossing_times_are_the_crossings_with_their_durations():
    pixels = np.array([[0.0, 0.0]])
    t = np.linspace(0, 20, 2001)  # 0.01 s
    xi = np.linspace(-1, 1, 2001)  # 0.1 deg/s
    idx, dur = crossing_times(t, xi, np.zeros_like(xi), pixels, 0.2)
    assert idx.tolist() == [0]
    assert dur[0] == pytest.approx(2 * 0.2 / 0.1, abs=0.02)  # 2 x radius / speed
    assert crossing_times(t, xi + 10, np.zeros_like(xi), pixels, 0.2)[0].size == 0


def test_the_same_as_gose_s_planner():
    gose = pytest.importorskip("gose.crossings")
    rng = np.random.default_rng(1)
    t = np.linspace(0, 420, 4000)
    tel_az, tel_el = 185 + 8 * np.sin(t / 30), np.full_like(t, 45.0)
    dr_az, dr_el = np.full_like(t, 185.0), 43 + 3 * (1 - np.cos(t / 40))
    a, b = (
        relative_path(tel_az, tel_el, dr_az, dr_el),
        gose.relative_path(tel_az, tel_el, dr_az, dr_el),
    )
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    pixels = rng.uniform(-10, 10, (500, 2))
    assert np.array_equal(crossings(*a, pixels, 0.25), gose.crossings(*b, pixels, 0.25))
    for x, y in zip(
        crossing_times(t, *a, pixels, 0.25),
        gose.crossing_times(t, *b, pixels, 0.25),
        strict=True,
    ):
        assert np.array_equal(x, y)
    assert np.array_equal(
        rotate(pixels[:, 0], pixels[:, 1], -30),
        gose.rotate(pixels[:, 0], pixels[:, 1], -30),
    )
