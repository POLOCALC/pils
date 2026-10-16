"""Tests for angle interpolation and drone records in the synchronizer."""

import numpy as np
import polars as pl
import pytest

from pils.synchronizer import Synchronizer, angle_period, interp_angle


class TestAnglePeriod:
    @pytest.mark.parametrize(
        "column",
        [
            "yaw",
            "roll",
            "pitch",
            "yaw(deg)",
            "RTK:yaw",
            "IMU_ATTI(0):yaw",
            "gimbalYaw",
            "gimbalYawAbs",
            "heading",
        ],
    )
    def test_degrees(self, column):
        assert angle_period(column) == 360.0

    def test_radians_and_raw(self):
        assert angle_period("yaw_rad") == pytest.approx(2 * np.pi)
        assert angle_period("gimbalYawRaw") == 3600.0
        assert angle_period("yawRaw") == 3600.0

    @pytest.mark.parametrize(
        "column",
        ["Controller:ctrl_yaw", "pqr_P_deg_s", "IMU_ATTI(0):gyroZ", "latitude"],
    )
    def test_not_angles(self, column):
        assert angle_period(column) is None


class TestInterpAngle:
    def test_crossing_180(self):
        out = interp_angle(np.array([0.5]), np.array([0.0, 1.0]), [179.0, -179.0])
        assert out[0] == pytest.approx(-180.0)

    def test_crossing_360(self):
        out = interp_angle(np.array([0.25]), np.array([0.0, 1.0]), [358.0, 2.0])
        assert out[0] == pytest.approx(359.0)

    def test_radians(self):
        xp = np.array([0.0, 1.0])
        out = interp_angle(np.array([0.5]), xp, [3.1, -3.1], period=2 * np.pi)
        assert abs(out[0]) == pytest.approx(np.pi)

    def test_nan_samples_and_outside(self):
        xp = np.arange(4.0)
        fp = np.array([170.0, np.nan, -170.0, -160.0])
        out = interp_angle(np.array([2.5, 5.0]), xp, fp)
        assert out[0] == pytest.approx(-165.0)
        assert np.isnan(out[1])


@pytest.fixture
def sync_with_drone():
    t = np.linspace(0, 100, 1000)
    position = {
        "latitude": 45.0 + 0.001 * np.sin(0.1 * t),
        "longitude": 10.0 + 0.001 * np.cos(0.1 * t),
        "altitude": 100.0 + 10.0 * np.sin(0.05 * t),
    }
    sync = Synchronizer()
    sync.add_gps_reference(
        pl.DataFrame({"timestamp": t, **position}),
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
    )
    # The drone clock is 0.5 s ahead; the yaw turns through +-180
    sync.add_drone_gps(
        pl.DataFrame(
            {
                "timestamp": t + 0.5,
                **position,
                "yaw": (t * 10.0 + 180.0) % 360.0 - 180.0,
            }
        ),
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
    )
    return sync


class TestSynchronize:
    def test_drone_yaw_across_the_wrap(self, sync_with_drone):
        result = sync_with_drone.synchronize(target_rate={"drone": 50.0})
        yaw = result["drone"]["yaw"].to_numpy()
        step = np.diff(yaw[np.isfinite(yaw)])
        step = (step + 180.0) % 360.0 - 180.0
        # 10 deg/s at 50 Hz: every step is 0.2 deg, also at the wrap
        np.testing.assert_allclose(step, 0.2, atol=1e-6)
        assert np.nanmin(yaw) >= -180.0 and np.nanmax(yaw) < 180.0

    def test_drone_records_take_drone_offset(self, sync_with_drone):
        t = np.arange(0, 101, 0.005)  # 200 Hz, drone clock
        sync_with_drone.add_drone_records(
            {
                "IMU_ATTI(0)": pl.DataFrame(
                    {
                        "tick": np.arange(t.size, dtype=np.int64),
                        "IMU_ATTI(0):pressure": t,
                        "IMU_ATTI(0):numSats": (t // 10).astype(np.int64),
                        "timestamp": t + 0.5,
                    }
                ),
                "no_time": pl.DataFrame({"tick": [1, 2]}),
            }
        )
        result = sync_with_drone.synchronize(
            target_rate={"drone": 10.0, "drone_records": 20.0}
        )
        assert "drone_no_time" not in result
        imu = result["drone_IMU_ATTI(0)"]
        offset = sync_with_drone.offsets["drone_gps"]["time_offset"]
        target = imu["timestamp"].to_numpy()
        assert target.size == 2001  # 100 s at 20 Hz
        # Back on the payload clock: drone time = target - offset
        drone_time = target - offset
        ok = np.isfinite(imu["IMU_ATTI(0):pressure"].to_numpy())
        np.testing.assert_allclose(
            imu["IMU_ATTI(0):pressure"].to_numpy()[ok],
            drone_time[ok] - 0.5,
            atol=1e-6,
        )
        # Integers hold the last value instead of being interpolated
        sats = imu["IMU_ATTI(0):numSats"].to_numpy()
        assert np.all(sats[np.isfinite(sats)] == np.floor(sats[np.isfinite(sats)]))

    def test_no_drone_offset_no_records(self):
        t = np.linspace(0, 10, 100)
        sync = Synchronizer()
        sync.add_gps_reference(
            pl.DataFrame({"timestamp": t, "lat": t, "lon": t, "alt": t}),
            lat_col="lat",
            lon_col="lon",
            alt_col="alt",
        )
        sync.add_drone_records({"IMU": pl.DataFrame({"timestamp": t, "yaw": t})})
        result = sync.synchronize(target_rate={"drone": 10.0})
        assert "drone_IMU" not in result


class TestResampleTable:
    def test_rules(self):
        from datetime import datetime

        from pils.synchronizer import resample_table

        t = np.array([0.0, 1.0, 2.0, 3.0])
        df = pl.DataFrame(
            {
                "timestamp": t,
                "unix_time_ms": (t * 1000).astype(np.int64),
                "datetime": [datetime(2025, 1, 1)] * 4,
                "note": ["a", "b", "c", "d"],
                "insStatus": pl.Series([1, 1, 7, 7], dtype=pl.Int64),
                "posllh_hMSL": pl.Series([0, 10, 20, 30], dtype=pl.Int64),
                "flag": [True, False, True, True],
                "yaw": [170.0, -170.0, -160.0, None],
            }
        )
        target = np.array([-1.0, 0.5, 1.5, 2.5, 4.0])
        out = resample_table(df, t, target)
        assert out.columns == ["timestamp", "insStatus", "posllh_hMSL", "flag", "yaw"]
        assert out["insStatus"].dtype == pl.Int64
        assert out["insStatus"].to_list() == [None, 1, 1, 7, None]
        assert out["posllh_hMSL"].to_list() == [None, 5.0, 15.0, 25.0, None]
        assert out["flag"].to_list() == [None, True, False, True, None]
        assert out["yaw"][1] == pytest.approx(-180.0)
        assert out["yaw"][3] is None  # no value in the source
        assert out["yaw"][0] is None  # outside the source

    def test_drops_rows_without_time_and_sorts(self):
        from pils.synchronizer import resample_table

        df = pl.DataFrame({"timestamp": [2.0, None, 0.0], "v": [20.0, 99.0, 0.0]})
        out = resample_table(
            df, df["timestamp"].to_numpy().astype(float), np.array([1.0])
        )
        assert out["v"].to_list() == [10.0]


def test_raw_and_sync_provenance(tmp_path):
    import h5py

    from pils.flight import Flight, PayloadData

    flight = Flight({"drone_data_folder_path": "", "aux_data_folder_path": ""})
    flight.raw_data.payload_data = PayloadData()
    flight.raw_data.payload_data.gps = pl.DataFrame(
        {
            "timestamp": np.arange(0.0, 20.0),
            "posllh_lat": np.full(20, 45.0),
            "posllh_lon": np.full(20, 10.0),
            "posllh_height": np.full(20, 100.0),
        }
    )
    synced = flight.sync()
    assert synced["reference_gps"].height == 20 * 100 - 99  # resampled at 100 Hz
    assert flight.sync_rates["reference_gps"] == 100.0
    path = tmp_path / "f.h5"
    flight.to_hdf5(path)
    with h5py.File(path) as f:
        saved_at = f["raw_data"].attrs["saved_at"]
        revision = f["sync_data"][sorted(f["sync_data"])[-1]]
        assert revision.attrs["raw_data_saved_at"] == saved_at
