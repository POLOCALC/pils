"""Tests for the shared analysis helpers: units, HDF5 input, uncertainty, geometry."""

import h5py
import numpy as np
import pytest

from pils.utils import geometry
from pils.utils.hdf5 import h5_to_dict, load_h5
from pils.utils.uncertainty import linear_interpolate_rows
from pils.utils.units import db_to_linear, dbm_to_mw, linear_to_db, mw_to_dbm


class TestUnits:
    def test_round_trip(self):
        values = np.array([-30.0, 0.0, 3.0, 20.0])
        np.testing.assert_allclose(linear_to_db(db_to_linear(values)), values)
        np.testing.assert_allclose(mw_to_dbm(dbm_to_mw(values)), values)

    def test_known_values(self):
        assert float(dbm_to_mw(0.0)) == pytest.approx(1.0)
        assert float(mw_to_dbm(0.001)) == pytest.approx(-30.0)


class TestHDF5Input:
    def test_h5_to_dict(self, tmp_path):
        path = tmp_path / "nested.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("a/b", data=[1, 2])
            f.create_dataset("c", data=3.0)
        data = load_h5(path)
        np.testing.assert_array_equal(data["a"]["b"], [1, 2])
        assert data["c"] == 3.0
        with h5py.File(path) as f:
            assert set(h5_to_dict(f)) == {"a", "c"}


class TestUncertainty:
    def test_linear_analytic(self):
        # Between two points, y = y0 + s (x - x0): var from x_new, y0, y1
        y, dy = linear_interpolate_rows(
            [[0.0, 1.0]], [[0.0, 2.0]], [[0.0, 0.0]], [[0.1, 0.1]], [0.5], [0.05]
        )
        assert y[0] == pytest.approx(1.0)
        expected = np.sqrt((2 * 0.05) ** 2 + 2 * (0.5 * 0.1) ** 2)
        assert dy[0] == pytest.approx(expected, rel=1e-6)

    def test_rows_are_independent_and_extrapolate(self):
        y, _ = linear_interpolate_rows(
            [[2.0, 1.0], [0.0, 1.0]],  # unsorted row
            [[4.0, 2.0], [0.0, -1.0]],
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            [1.5, 2.0],
            [0.0, 0.0],
        )
        np.testing.assert_allclose(y, [3.0, -2.0])


class TestGeometry:
    def test_poi_straight_down(self):
        poi = geometry.compute_poi([10.0], [20.0], [100.0], [0.0], [-90.0], [0.0])
        np.testing.assert_allclose(poi[0], [10.0, 20.0, 0.0], atol=1e-9)

    def test_poi_45_deg_north_and_up_is_nan(self):
        poi = geometry.compute_poi(
            [0.0, 0.0], [0.0, 0.0], [100.0, 100.0], [0.0, 0.0], [-45.0, 10.0], [0, 0]
        )
        np.testing.assert_allclose(poi[0], [0.0, 100.0, 0.0], atol=1e-9)
        assert np.isnan(poi[1]).all()

    def test_poi_yaw_clockwise(self):
        # Yaw 90° clockwise from North points East
        poi = geometry.compute_poi([0.0], [0.0], [100.0], [90.0], [-45.0], [0.0])
        np.testing.assert_allclose(poi[0], [100.0, 0.0, 0.0], atol=1e-9)

    def test_beam_fwhm(self):
        x = np.linspace(-10, 10, 2001)
        beam = geometry.beam_2d_gauss(x, x, fwhm_h=4.0, fwhm_e=2.0)
        centre = len(x) // 2
        assert beam[centre, centre] == pytest.approx(1.0)
        # Half power at half the FWHM on each plane
        assert beam[centre, np.argmin(abs(x - 2.0))] == pytest.approx(0.5, abs=1e-3)
        assert beam[np.argmin(abs(x - 1.0)), centre] == pytest.approx(0.5, abs=1e-3)

    def test_frame_and_relative_angles(self):
        transform = geometry.build_frame([0, 0, 100], [100, 0, 0], roll_deg=0.0)
        local_target = geometry.world_to_local(transform, [100, 0, 0])
        assert local_target[0] == pytest.approx(np.hypot(100, 100))
        np.testing.assert_allclose(local_target[1:], [0, 0], atol=1e-9)
        assert geometry.relative_yaw_pitch_roll([1, 0, 0], [1, 0, 0]) == (0, 0, 0)
        yaw, pitch, _ = geometry.relative_yaw_pitch_roll([1, 0, 0], [0, 1, 0])
        assert yaw == pytest.approx(np.pi / 2)
        assert pitch == pytest.approx(0.0)


def test_new_version_name_is_utc():
    """A revision is named in UTC (STOUT's layout), a naive datetime taken as UTC."""
    import datetime as dt

    from pils import store

    chile = dt.timezone(-dt.timedelta(hours=3))
    assert store.new_version_name(
        dt.datetime(2025, 12, 8, 16, 27, 5, tzinfo=chile)
    ) == ("rev_20251208_192705")
    assert (
        store.new_version_name(dt.datetime(2025, 12, 8, 19, 27, 5))
        == "rev_20251208_192705"
    )


def test_so_books_beside_campaigns_for_a_field_test(tmp_path):
    """SO's books are in campaigns/telescope_data whether the flight is in campaigns/ or field_tests/."""
    from pils.telescope_pointing import so_data_root

    for rel in (
        "campaigns/202511/20251208/flight_20251208_1927",
        "field_tests/gimbal_test/20250301/flight_20250301_1000",
    ):
        (tmp_path / rel).mkdir(parents=True)
        assert so_data_root(tmp_path / rel) == (
            tmp_path.resolve() / "campaigns" / "telescope_data" / "SO" / "data"
        )
