"""Positions on the site, and how a telescope sees the drone (pils.geometry.site)."""

import numpy as np
import pytest

from pils.geometry.site import (
    azel_to_unit,
    circular_mean_deg,
    direction_from,
    enu_from_distance,
    enu_from_reference,
    enu_to_azel,
    enu_to_geodetic,
    seen_from,
    site_enu,
)

pytest.importorskip("so3g")


def test_the_reference_site_is_the_origin_and_the_sats_are_near_it():
    assert np.allclose(site_enu("so_lat"), 0)
    sat1 = site_enu("so_sat1")
    assert 50 < np.hypot(sat1[0], sat1[1]) < 200 and sat1[2] == 0  # all at 5188 m
    assert np.array_equal(site_enu((1.0, 2.0, 3.0)), [1.0, 2.0, 3.0])


def test_geodetic_is_the_inverse_of_the_site_frame():
    from so3g.proj import coords

    lat, lon, h = enu_to_geodetic(np.array([site_enu("so_sat3")]))
    s = coords.SITES["so_sat3"]
    assert lat[0] == pytest.approx(s.lat, abs=1e-9) and lon[0] == pytest.approx(
        s.lon, abs=1e-9
    )
    assert h[0] == pytest.approx(s.elev)


def test_directions_go_both_ways():
    v = 480 * azel_to_unit(185.0, 45.0)
    az, el, rng = enu_to_azel(v)
    assert (az, el, rng) == (
        pytest.approx(185.0),
        pytest.approx(45.0),
        pytest.approx(480.0),
    )
    point = enu_from_distance("so_sat3", 185.0, 45.0, 480.0)
    assert direction_from("so_sat3", point) == (
        pytest.approx(185.0),
        pytest.approx(45.0),
        pytest.approx(480.0),
    )
    up = enu_from_reference("so_sat3", 185.0, 45.0, 300.0)
    assert up[2] - site_enu("so_sat3")[2] == pytest.approx(300.0)


def test_a_drone_seen_from_a_site_is_pils_azel_columns():
    track = np.array(
        [enu_from_distance("so_sat3", 185.0, el, 480.0) for el in (40.0, 45.0)]
    )
    seen = seen_from(track, site_enu("so_sat3"))
    assert set(seen) == {"az_deg", "el_deg", "srange_m", "e_m", "n_m", "u_m"}
    assert np.allclose(seen["el_deg"], [40.0, 45.0]) and np.allclose(
        seen["srange_m"], 480.0
    )


def test_azimuths_average_on_the_circle():
    m = circular_mean_deg([350.0, 10.0])  # north, not 180
    assert min(m, 360.0 - m) == pytest.approx(0.0, abs=1e-9)


def test_the_same_as_gose_s_planner():
    gose = pytest.importorskip("gose.sat_pointing")
    for key in ("so_sat1", "so_sat2", "so_sat3", "so_lat"):
        assert np.array_equal(site_enu(key), gose.site_enu(key))
    p = np.array([120.0, -340.0, 310.0])
    assert direction_from("so_sat1", p) == gose.direction_from("so_sat1", p)
    assert np.array_equal(
        enu_from_reference("so_sat3", 185, 45, 300),
        gose.enu_from_reference("so_sat3", 185, 45, 300),
    )
    assert np.array_equal(
        enu_from_distance("so_sat3", 185, 45, 480),
        gose.enu_from_distance("so_sat3", 185, 45, 480),
    )
    assert circular_mean_deg([350, 20, 5]) == gose.circular_mean_deg([350, 20, 5])
