"""Tests for the shared analysis helpers: units, HDF5 versions, uncertainty, geometry."""

import h5py
import numpy as np
import polars as pl
import pytest

from pils.utils import geometry
from pils.utils.hdf5 import (
    h5_to_dict,
    list_versions,
    load_h5,
    load_version,
    new_version_name,
    save_version,
)
from pils.utils.uncertainty import (
    interpolate_with_errors,
    interpolate_with_errors_jacobian,
    mc_interp_sigma,
)
from pils.utils.units import db_to_linear, dbm_to_mw, linear_to_db, mw_to_dbm


class TestUnits:
    def test_round_trip(self):
        values = np.array([-30.0, 0.0, 3.0, 20.0])
        np.testing.assert_allclose(linear_to_db(db_to_linear(values)), values)
        np.testing.assert_allclose(mw_to_dbm(dbm_to_mw(values)), values)

    def test_known_values(self):
        assert float(dbm_to_mw(0.0)) == pytest.approx(1.0)
        assert float(mw_to_dbm(0.001)) == pytest.approx(-30.0)


class TestVersionedHDF5:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "proc" / "x" / "x_solution.h5"
        frame = pl.DataFrame(
            {"freq": [84.0, 87.0], "count": [1, 2], "plane": ["H", "E"]}
        )
        save_version(path, "rev_20260101_120000", {"gain": frame}, {"n": np.int64(3)})
        tables, metadata = load_version(path, "rev_20260101_120000")
        assert tables["gain"].equals(frame)
        assert metadata == {"n": 3}

    def test_versions_with_parent(self, tmp_path):
        path = tmp_path / "a.h5"
        frame = pl.DataFrame({"x": [1.0]})
        save_version(path, "rev_20260101_120000", {"t": frame}, {}, parent="SATP1")
        save_version(path, "rev_20260102_120000", {"t": frame}, {}, parent="SATP1")
        save_version(path, "rev_20260103_120000", {"t": frame}, {}, parent="SATP3")
        assert list_versions(path, parent="SATP1") == [
            "rev_20260101_120000",
            "rev_20260102_120000",
        ]
        assert list_versions(path, parent="NOPE") == []
        assert list_versions(tmp_path / "missing.h5") == []
        with h5py.File(path) as f:
            assert f.attrs["latest_version"] == "rev_20260103_120000"

    def test_missing_version(self, tmp_path):
        path = tmp_path / "a.h5"
        save_version(path, "rev_20260101_120000", {}, {})
        with pytest.raises(KeyError):
            load_version(path, "rev_20990101_000000")
        with pytest.raises(FileNotFoundError):
            load_version(tmp_path / "missing.h5", "rev_20260101_120000")

    def test_version_name_format(self):
        assert new_version_name().startswith("rev_")
        assert len(new_version_name()) == len("rev_20260101_120000")

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
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    Y = np.array([2.1, 4.3, 6.2, 8.1, 10.0])
    DX = np.full(5, 0.1)
    DY = np.full(5, 0.2)

    def test_linear_analytic(self):
        # Between two points, y = y0 + s (x - x0): var from x_new, y0, y1, x0, x1
        y, dy = interpolate_with_errors(
            [0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [0.1, 0.1], 0.5, 0.05
        )
        assert y == pytest.approx(1.0)
        expected = np.sqrt((2 * 0.05) ** 2 + 2 * (0.5 * 0.1) ** 2)
        assert dy == pytest.approx(expected, rel=1e-6)

    def test_vectorised_matches_single(self):
        x_new = np.array([1.5, 2.5, 3.5])
        dx_new = np.array([0.1, 0.15, 0.12])
        y, dy = interpolate_with_errors_jacobian(
            self.X, self.Y, self.DX, self.DY, x_new, dx_new
        )
        for i in range(3):
            yi, dyi = interpolate_with_errors(
                self.X, self.Y, self.DX, self.DY, x_new[i], dx_new[i]
            )
            assert y[i] == pytest.approx(yi)
            assert dy[i] == pytest.approx(dyi, rel=1e-6)

    def test_monte_carlo_agrees(self):
        _, dy = interpolate_with_errors(self.X, self.Y, self.DX, self.DY, 2.5, 0.15)
        _, dy_mc = mc_interp_sigma(
            self.X,
            self.Y,
            self.DX,
            self.DY,
            2.5,
            0.15,
            n=4000,
            rng=np.random.default_rng(1),
        )
        assert dy_mc == pytest.approx(dy, rel=0.1)


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
