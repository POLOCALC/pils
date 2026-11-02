"""AZEL's RTK base correction: its sign, and when it applies.

A DJI RTK rover solves its position from the base position the base
broadcasts. With the broadcast position off the surveyed one by
``delta = true_base - broadcast_base``, the rover reports
``true_rover - delta``: AZEL must add ``delta`` back.
"""

import numpy as np
import polars as pl
import pymap3d as pm
import pytest

from pils.analyze.azel import AZELAnalysis
from pils.flight import DroneData, Flight

TELESCOPE = {"lat": -22.96009, "lon": -67.78838, "alt": 5184.89}
TRUE_BASE = {"lat": -22.95977498, "lon": -67.78669706, "alt": 5182.214}
# The 202511 broadcast base: ~9.2 m below and ~1.3 m east of the survey
BROADCAST = {"lat": -22.9597732, "lon": -67.7866847, "alt": 5173.02}


@pytest.fixture
def campaign(tmp_path):
    flight_dir = tmp_path / "campaigns" / "flight_test"
    (flight_dir / "drone").mkdir(parents=True)
    (tmp_path / "metadata").mkdir()
    pl.DataFrame(
        {
            "Name": ["SATP1", "dji rtk base (antenna base)"],
            "Latitude": [TELESCOPE["lat"], TRUE_BASE["lat"]],
            "Longitude": [TELESCOPE["lon"], TRUE_BASE["lon"]],
            "Ellipsoidal height": [TELESCOPE["alt"], TRUE_BASE["alt"]],
        }
    ).write_csv(tmp_path / "metadata" / "202511_coordinates.csv")
    return flight_dir


def _true_track(n=50):
    """A drone track around the telescope: east/north/up from it (m)."""
    t = np.linspace(0, 2 * np.pi, n)
    e, n_, u = 300 * np.cos(t), 300 * np.sin(t), 150 + 20 * np.sin(3 * t)
    return e, n_, u


def _reported(e, n, u):
    """The track as a rover referenced to the broadcast base reports it."""
    x, y, z = pm.enu2ecef(e, n, u, TELESCOPE["lat"], TELESCOPE["lon"], TELESCOPE["alt"])
    bx, by, bz = pm.geodetic2ecef(*TRUE_BASE.values())
    cx, cy, cz = pm.geodetic2ecef(*BROADCAST.values())
    # true_rover - (true_base - broadcast_base)
    return pm.ecef2geodetic(x - (bx - cx), y - (by - cy), z - (bz - cz))


def _flight(campaign, table, sync=False, position_columns=None):
    flight = Flight({"drone_data_folder_path": str(campaign / "drone")})
    if sync:
        flight.sync_data = {"drone": table}
        flight.sync_position_columns = position_columns or {}
    else:
        flight.raw_data.drone_data = DroneData(drone_df=table, litchi_df=None)
    flight._Flight__drone_model = "dji"
    return flight


def _expected_azel(e, n, u):
    az, el, _ = pm.enu2aer(e, n, u)
    return az, el


def test_correction_recovers_the_true_azel_raw(campaign):
    e, n, u = _true_track()
    lat, lon, h = _reported(e, n, u)
    table = pl.DataFrame(
        {
            "timestamp": 1.76e9 + np.arange(e.size, dtype=float),
            "rtk_lat_deg": lat,
            "rtk_lon_deg": lon,
            "rtk_h_ell_m": h,
            "rtk_pos_flag_1": np.full(e.size, 50),
        }
    )
    version = AZELAnalysis(_flight(campaign, table)).run_analysis("SATP1", BROADCAST)
    assert version is not None
    assert version.metadata["rtk_applied"]
    # The broadcast base is ~9.19 m below the surveyed one
    assert version.metadata["rtk_correction"]["delta_u"] == pytest.approx(
        9.19, abs=0.01
    )
    az, el = _expected_azel(e, n, u)
    data = version.azel_data
    np.testing.assert_allclose(data["e_m"].to_numpy(), e, atol=1e-3)
    np.testing.assert_allclose(data["n_m"].to_numpy(), n, atol=1e-3)
    np.testing.assert_allclose(data["u_m"].to_numpy(), u, atol=1e-3)
    np.testing.assert_allclose(data["el_deg"].to_numpy(), el, atol=1e-5)
    daz = (data["az_deg"].to_numpy() - az + 180) % 360 - 180
    np.testing.assert_allclose(daz, 0, atol=1e-5)


def test_uncorrected_track_is_off_by_the_base_error(campaign):
    """Without the base, the error is the broadcast error (and the old
    subtraction doubled it): a check the test above is not trivial."""
    e, n, u = _true_track()
    lat, lon, h = _reported(e, n, u)
    table = pl.DataFrame(
        {
            "timestamp": 1.76e9 + np.arange(e.size, dtype=float),
            "rtk_lat_deg": lat,
            "rtk_lon_deg": lon,
            "rtk_h_ell_m": h,
        }
    )
    reported_u = pm.geodetic2enu(lat, lon, h, *TELESCOPE.values())[2]
    assert np.median(u - reported_u) == pytest.approx(9.19, abs=0.05)
    version = AZELAnalysis(_flight(campaign, table)).run_analysis("SATP1", BROADCAST)
    np.testing.assert_allclose(version.azel_data["u_m"].to_numpy(), u, atol=1e-3)


def test_sync_rtk_positions_corrected_and_fixed_only(campaign):
    e, n, u = _true_track()
    lat, lon, h = _reported(e, n, u)
    flags = np.full(e.size, 50)
    flags[:5] = 34  # float
    flags[5:8] = 16  # single
    table = pl.DataFrame(
        {
            "timestamp": 1.76e9 + np.arange(e.size, dtype=float),
            "lat_deg": lat,
            "lon_deg": lon,
            "h_ell_m": h,
            "rtk_pos_flag_1": flags,
        }
    )
    columns = {
        "drone": {"lat": "rtk_lat_deg", "lon": "rtk_lon_deg", "alt": "rtk_h_ell_m"}
    }
    version = AZELAnalysis(_flight(campaign, table, True, columns)).run_analysis(
        "SATP1", BROADCAST
    )
    assert version.metadata["rtk_applied"]
    assert version.metadata["rtk_fixed_only"]
    assert version.azel_data.height == e.size - 8
    np.testing.assert_allclose(version.azel_data["u_m"].to_numpy(), u[8:], atol=1e-3)


def test_sync_drone_gnss_not_corrected_nor_flag_filtered(campaign):
    e, n, u = _true_track()
    lat, lon, h = pm.enu2geodetic(e, n, u, *TELESCOPE.values())
    flags = np.full(e.size, 0)  # no RTK at all: the flag says nothing of GNSS
    table = pl.DataFrame(
        {
            "timestamp": 1.76e9 + np.arange(e.size, dtype=float),
            "lat_deg": lat,
            "lon_deg": lon,
            "h_ell_m": h,
            "rtk_lat_deg": np.zeros(e.size),
            "rtk_pos_flag_1": flags,
        }
    )
    columns = {
        "drone": {"lat": "gps_lat_deg", "lon": "gps_lon_deg", "alt": "gps_h_ell_m"}
    }
    version = AZELAnalysis(_flight(campaign, table, True, columns)).run_analysis(
        "SATP1", BROADCAST
    )
    assert not version.metadata["rtk_applied"]
    assert not version.metadata["rtk_fixed_only"]
    assert version.metadata["rtk_correction"]["delta_u"] == 0.0
    assert version.azel_data.height == e.size
    np.testing.assert_allclose(version.azel_data["u_m"].to_numpy(), u, atol=1e-3)


@pytest.mark.parametrize(
    "columns, rtk",
    [
        # RTK latitude renamed to lat_deg, other rtk_* columns kept
        (["lat_deg", "lon_deg", "h_ell_m", "rtk_pos_flag_1"], True),
        # The drone GNSS used: the RTK latitude still there
        (["lat_deg", "lon_deg", "h_ell_m", "rtk_lat_deg"], False),
        # The GNSS record of a .dat file: no RTK column at all
        (["lat_deg", "lon_deg", "h_ell_m"], False),
    ],
)
def test_a_sync_without_the_record_judged_from_its_columns(columns, rtk):
    from pils.analyze.azel import _positions_are_rtk

    table = pl.DataFrame({c: [1.0] for c in columns})
    assert _positions_are_rtk("sync_data", table, None) is rtk
    assert _positions_are_rtk("raw_data", table, None) is True


def test_broadcast_reference_keeps_the_positions_as_reported(campaign):
    """rtk_reference="broadcast": the frame of the broadcast base, no correction."""
    e, n, u = _true_track()
    lat, lon, h = _reported(e, n, u)
    table = pl.DataFrame(
        {
            "timestamp": 1.76e9 + np.arange(e.size, dtype=float),
            "rtk_lat_deg": lat,
            "rtk_lon_deg": lon,
            "rtk_h_ell_m": h,
            "rtk_pos_flag_1": np.full(e.size, 50),
        }
    )
    version = AZELAnalysis(_flight(campaign, table)).run_analysis(
        "SATP1", BROADCAST, rtk_reference="broadcast"
    )
    assert version is not None
    assert not version.metadata["rtk_applied"]
    assert version.metadata["rtk_reference"] == "broadcast"
    assert version.metadata["rtk_fixed_only"]  # still RTK positions
    reported = pm.geodetic2enu(lat, lon, h, *TELESCOPE.values())
    for column, expected in zip(("e_m", "n_m", "u_m"), reported):
        np.testing.assert_allclose(version.azel_data[column].to_numpy(), expected, atol=1e-3)


def test_unknown_rtk_reference_refused(campaign):
    table = pl.DataFrame({"timestamp": [1.76e9], "rtk_lat_deg": [0.0]})
    with pytest.raises(ValueError, match="rtk_reference"):
        AZELAnalysis(_flight(campaign, table)).run_analysis(
            "SATP1", BROADCAST, rtk_reference="emlid"
        )
