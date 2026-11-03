"""
Tests for coordinate column name standardization across drone types.

Verifies that all GPS sources output standardized column names:
- latitude, longitude, altitude, timestamp
regardless of input column names (RTK:lat_p, GPS:Latitude, etc.)
"""

import numpy as np
import polars as pl
import pytest


class TestCoordinateStandardization:
    """Test standardized coordinate column names in sync output."""

    @pytest.fixture
    def sample_gps_payload(self):
        """Create GPS payload data with u-blox column names."""
        t = np.linspace(0, 100, 1000)
        data = pl.DataFrame(
            {
                "timestamp": t,
                "lat_deg": 45.0 + 0.001 * np.sin(0.1 * t),
                "lon_deg": 10.0 + 0.001 * np.cos(0.1 * t),
                "h_ell_m": 100.0 + 10.0 * np.sin(0.05 * t),
            }
        )
        return data

    @pytest.fixture
    def sample_drone_dji_rtk(self):
        """Create DJI drone data with RTK column names."""
        t = np.linspace(0, 100, 1000)
        data = pl.DataFrame(
            {
                "timestamp": t,
                "rtk_lat_deg": 45.0 + 0.001 * np.sin(0.1 * (t + 0.5)),
                "rtk_lon_deg": 10.0 + 0.001 * np.cos(0.1 * (t + 0.5)),
                "rtk_h_ell_m": 100.0 + 10.0 * np.sin(0.05 * (t + 0.5)),
                "battery_percent": 85.0,
                "gimbal_pitch_deg": -45.0,
            }
        )
        return data

    @pytest.fixture
    def sample_drone_dji_gps(self):
        """Create DJI drone data with standard GPS column names."""
        t = np.linspace(0, 100, 1000)
        data = pl.DataFrame(
            {
                "timestamp": t,
                "GPS:Latitude": 45.0 + 0.001 * np.sin(0.1 * (t + 0.5)),
                "GPS:Longitude": 10.0 + 0.001 * np.cos(0.1 * (t + 0.5)),
                "GPS:heightMSL": 100.0 + 10.0 * np.sin(0.05 * (t + 0.5)),
                "battery_percent": 85.0,
                "gimbal_pitch_deg": -45.0,
            }
        )
        return data

    @pytest.fixture
    def sample_drone_blacksquare(self):
        """Create BlackSquare drone data."""
        t = np.linspace(0, 100, 1000)
        data = pl.DataFrame(
            {
                "timestamp": t,
                "lat_deg": 45.0 + 0.001 * np.sin(0.1 * (t + 0.5)),
                "lon_deg": 10.0 + 0.001 * np.cos(0.1 * (t + 0.5)),
                "h_msl_m": 100.0 + 10.0 * np.sin(0.05 * (t + 0.5)),
                "voltage": 12.5,
            }
        )
        return data

    @pytest.fixture
    def sample_litchi_gps(self):
        """Create Litchi GPS data."""
        t = np.linspace(0, 100, 1000)
        data = pl.DataFrame(
            {
                "timestamp": t,
                "lat_deg": 45.0 + 0.001 * np.sin(0.1 * (t + 0.3)),
                "lon_deg": 10.0 + 0.001 * np.cos(0.1 * (t + 0.3)),
                "h_rel_m": 100.0 + 10.0 * np.sin(0.05 * (t + 0.3)),
                "gimbal_pitch_deg": -60.0,
            }
        )
        return data

    def test_drone_rtk_coordinates_standardized(
        self, sample_gps_payload, sample_drone_dji_rtk
    ):
        """DJI RTK outputs standard latitude/longitude/altitude columns."""
        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.add_gps_reference(
            sample_gps_payload,
            timestamp_col="timestamp",
            lat_col="lat_deg",
            lon_col="lon_deg",
            alt_col="h_ell_m",
        )
        sync.add_drone_gps(
            sample_drone_dji_rtk,
            timestamp_col="timestamp",
            lat_col="rtk_lat_deg",
            lon_col="rtk_lon_deg",
            alt_col="rtk_h_ell_m",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        assert "drone" in result
        drone_df = result["drone"]

        # Verify standardized column names exist
        assert "lat_deg" in drone_df.columns, (
            "Should have standardized 'lat_deg' column"
        )
        assert "lon_deg" in drone_df.columns, (
            "Should have standardized 'lon_deg' column"
        )
        assert "h_ell_m" in drone_df.columns, (
            "Should have standardized 'h_ell_m' column"
        )
        assert "timestamp" in drone_df.columns, (
            "Should have standardized 'timestamp' column"
        )

        # Verify original column names do NOT exist
        assert "rtk_lat_deg" not in drone_df.columns, (
            "Original 'rtk_lat_deg' should be renamed"
        )
        assert "rtk_lon_deg" not in drone_df.columns, (
            "Original 'rtk_lon_deg' should be renamed"
        )
        assert "rtk_h_ell_m" not in drone_df.columns, (
            "Original 'rtk_h_ell_m' should be renamed"
        )
        # timestamp is already the standard name after DJI loader changes
        # No need to check for old column name

        # Verify other columns are preserved
        assert "battery_percent" in drone_df.columns, (
            "Non-coordinate columns should be preserved"
        )
        assert "gimbal_pitch_deg" in drone_df.columns, (
            "Non-coordinate columns should be preserved"
        )

    def test_drone_gps_coordinates_standardized(
        self, sample_gps_payload, sample_drone_dji_gps
    ):
        """DJI standard GPS outputs standard latitude/longitude/altitude columns."""
        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.add_gps_reference(
            sample_gps_payload,
            timestamp_col="timestamp",
            lat_col="lat_deg",
            lon_col="lon_deg",
            alt_col="h_ell_m",
        )
        sync.add_drone_gps(
            sample_drone_dji_gps,
            timestamp_col="timestamp",
            lat_col="GPS:Latitude",
            lon_col="GPS:Longitude",
            alt_col="GPS:heightMSL",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        drone_df = result["drone"]

        # Verify standardized column names
        assert "lat_deg" in drone_df.columns
        assert "lon_deg" in drone_df.columns
        assert "h_ell_m" in drone_df.columns
        assert "timestamp" in drone_df.columns

        # Verify original names removed
        assert "GPS:Latitude" not in drone_df.columns
        assert "GPS:Longitude" not in drone_df.columns
        assert "GPS:heightMSL" not in drone_df.columns

    def test_drone_blacksquare_coordinates_standardized(
        self, sample_gps_payload, sample_drone_blacksquare
    ):
        """BlackSquare outputs standard latitude/longitude/altitude columns."""
        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.add_gps_reference(
            sample_gps_payload,
            timestamp_col="timestamp",
            lat_col="lat_deg",
            lon_col="lon_deg",
            alt_col="h_ell_m",
        )
        sync.add_drone_gps(
            sample_drone_blacksquare,
            timestamp_col="timestamp",
            lat_col="lat_deg",
            lon_col="lon_deg",
            alt_col="h_msl_m",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        drone_df = result["drone"]

        # Verify standardized column names
        assert "lat_deg" in drone_df.columns
        assert "lon_deg" in drone_df.columns
        assert "h_ell_m" in drone_df.columns
        assert "timestamp" in drone_df.columns

        # The height column given as the ellipsoidal one is renamed
        assert "h_msl_m" not in drone_df.columns

        # Verify other columns preserved
        assert "voltage" in drone_df.columns

    def test_litchi_coordinates_standardized(
        self, sample_gps_payload, sample_litchi_gps
    ):
        """Litchi outputs standard latitude/longitude/altitude columns."""
        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.add_gps_reference(
            sample_gps_payload,
            timestamp_col="timestamp",
            lat_col="lat_deg",
            lon_col="lon_deg",
            alt_col="h_ell_m",
        )
        sync.add_litchi_gps(sample_litchi_gps)

        result = sync.synchronize(target_rate={"drone": 10.0})

        assert "litchi" in result
        litchi_df = result["litchi"]

        # Verify standardized column names (litchi already has lowercase latitude/longitude)
        assert "lat_deg" in litchi_df.columns
        assert "lon_deg" in litchi_df.columns
        assert "h_ell_m" in litchi_df.columns
        assert "timestamp" in litchi_df.columns

        # The height above take-off is kept, h_ell_m is added
        assert "h_rel_m" in litchi_df.columns

    def test_payload_gps_coordinates_standardized(self, sample_gps_payload):
        """Payload GPS outputs standard latitude/longitude/altitude columns."""
        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.add_gps_reference(
            sample_gps_payload,
            timestamp_col="timestamp",
            lat_col="lat_deg",
            lon_col="lon_deg",
            alt_col="h_ell_m",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        assert "reference_gps" in result
        gps_df = result["reference_gps"]

        # Verify standardized column names
        assert "lat_deg" in gps_df.columns
        assert "lon_deg" in gps_df.columns
        assert "h_ell_m" in gps_df.columns
        assert "timestamp" in gps_df.columns

    def test_all_gps_sources_same_column_names(
        self, sample_gps_payload, sample_drone_dji_rtk, sample_litchi_gps
    ):
        """All GPS sources have identical coordinate column names."""
        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.add_gps_reference(
            sample_gps_payload,
            timestamp_col="timestamp",
            lat_col="lat_deg",
            lon_col="lon_deg",
            alt_col="h_ell_m",
        )
        sync.add_drone_gps(
            sample_drone_dji_rtk,
            timestamp_col="timestamp",
            lat_col="rtk_lat_deg",
            lon_col="rtk_lon_deg",
            alt_col="rtk_h_ell_m",
        )
        sync.add_litchi_gps(sample_litchi_gps)

        result = sync.synchronize(target_rate={"drone": 10.0})

        # Get coordinate columns from each source
        drone_coords = set(result["drone"].columns) & {
            "lat_deg",
            "lon_deg",
            "h_ell_m",
            "timestamp",
        }
        litchi_coords = set(result["litchi"].columns) & {
            "lat_deg",
            "lon_deg",
            "h_ell_m",
            "timestamp",
        }
        ref_coords = set(result["reference_gps"].columns) & {
            "lat_deg",
            "lon_deg",
            "h_ell_m",
            "timestamp",
        }

        # All sources should have the same standard coordinate columns
        assert drone_coords == {"lat_deg", "lon_deg", "h_ell_m", "timestamp"}
        assert litchi_coords == {"lat_deg", "lon_deg", "h_ell_m", "timestamp"}
        assert ref_coords == {"lat_deg", "lon_deg", "h_ell_m", "timestamp"}

    def test_coordinate_values_unchanged(
        self, sample_gps_payload, sample_drone_dji_rtk
    ):
        """Verify renaming doesn't change coordinate values."""
        import polars as pl

        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.add_gps_reference(
            sample_gps_payload,
            timestamp_col="timestamp",
            lat_col="lat_deg",
            lon_col="lon_deg",
            alt_col="h_ell_m",
        )
        sync.add_drone_gps(
            sample_drone_dji_rtk,
            timestamp_col="timestamp",
            lat_col="rtk_lat_deg",
            lon_col="rtk_lon_deg",
            alt_col="rtk_h_ell_m",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        drone_df = result["drone"]

        # Values should be reasonable (filter out NaN from interpolation edges)
        # Polars drop_nulls() doesn't remove NaN, so filter explicitly
        valid_lat = drone_df.filter(pl.col("lat_deg").is_not_nan())["lat_deg"]
        valid_lon = drone_df.filter(pl.col("lon_deg").is_not_nan())["lon_deg"]
        valid_alt = drone_df.filter(pl.col("h_ell_m").is_not_nan())["h_ell_m"]

        assert valid_lat.mean() > 44.5
        assert valid_lat.mean() < 45.5
        assert valid_lon.mean() > 9.5
        assert valid_lon.mean() < 10.5
        assert valid_alt.mean() > 90.0
        assert valid_alt.mean() < 110.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
