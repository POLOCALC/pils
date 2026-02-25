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
                "posllh_lat": 45.0 + 0.001 * np.sin(0.1 * t),
                "posllh_lon": 10.0 + 0.001 * np.cos(0.1 * t),
                "posllh_height": 100.0 + 10.0 * np.sin(0.05 * t),
            }
        )
        return data

    @pytest.fixture
    def sample_drone_dji_rtk(self):
        """Create DJI drone data with RTK column names."""
        t = np.linspace(0, 100, 1000)
        data = pl.DataFrame(
            {
                "correct_timestamp": t,
                "RTK:lat_p": 45.0 + 0.001 * np.sin(0.1 * (t + 0.5)),
                "RTK:lon_p": 10.0 + 0.001 * np.cos(0.1 * (t + 0.5)),
                "RTK:hmsl_p": 100.0 + 10.0 * np.sin(0.05 * (t + 0.5)),
                "battery_percent": 85.0,
                "gimbalPitch": -45.0,
            }
        )
        return data

    @pytest.fixture
    def sample_drone_dji_gps(self):
        """Create DJI drone data with standard GPS column names."""
        t = np.linspace(0, 100, 1000)
        data = pl.DataFrame(
            {
                "correct_timestamp": t,
                "GPS:Latitude": 45.0 + 0.001 * np.sin(0.1 * (t + 0.5)),
                "GPS:Longitude": 10.0 + 0.001 * np.cos(0.1 * (t + 0.5)),
                "GPS:heightMSL": 100.0 + 10.0 * np.sin(0.05 * (t + 0.5)),
                "battery_percent": 85.0,
                "gimbalPitch": -45.0,
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
                "Latitude": 45.0 + 0.001 * np.sin(0.1 * (t + 0.5)),
                "Longitude": 10.0 + 0.001 * np.cos(0.1 * (t + 0.5)),
                "heightMSL": 100.0 + 10.0 * np.sin(0.05 * (t + 0.5)),
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
                "latitude": 45.0 + 0.001 * np.sin(0.1 * (t + 0.3)),
                "longitude": 10.0 + 0.001 * np.cos(0.1 * (t + 0.3)),
                "altitude(m)": 100.0 + 10.0 * np.sin(0.05 * (t + 0.3)),
                "gimbalPitch": -60.0,
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
            lat_col="posllh_lat",
            lon_col="posllh_lon",
            alt_col="posllh_height",
        )
        sync.add_drone_gps(
            sample_drone_dji_rtk,
            timestamp_col="correct_timestamp",
            lat_col="RTK:lat_p",
            lon_col="RTK:lon_p",
            alt_col="RTK:hmsl_p",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        assert "drone" in result
        drone_df = result["drone"]

        # Verify standardized column names exist
        assert "latitude" in drone_df.columns, (
            "Should have standardized 'latitude' column"
        )
        assert "longitude" in drone_df.columns, (
            "Should have standardized 'longitude' column"
        )
        assert "altitude" in drone_df.columns, (
            "Should have standardized 'altitude' column"
        )
        assert "timestamp" in drone_df.columns, (
            "Should have standardized 'timestamp' column"
        )

        # Verify original column names do NOT exist
        assert "RTK:lat_p" not in drone_df.columns, (
            "Original 'RTK:lat_p' should be renamed"
        )
        assert "RTK:lon_p" not in drone_df.columns, (
            "Original 'RTK:lon_p' should be renamed"
        )
        assert "RTK:hmsl_p" not in drone_df.columns, (
            "Original 'RTK:hmsl_p' should be renamed"
        )
        assert "correct_timestamp" not in drone_df.columns, (
            "Original 'correct_timestamp' should be renamed"
        )

        # Verify other columns are preserved
        assert "battery_percent" in drone_df.columns, (
            "Non-coordinate columns should be preserved"
        )
        assert "gimbalPitch" in drone_df.columns, (
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
            lat_col="posllh_lat",
            lon_col="posllh_lon",
            alt_col="posllh_height",
        )
        sync.add_drone_gps(
            sample_drone_dji_gps,
            timestamp_col="correct_timestamp",
            lat_col="GPS:Latitude",
            lon_col="GPS:Longitude",
            alt_col="GPS:heightMSL",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        drone_df = result["drone"]

        # Verify standardized column names
        assert "latitude" in drone_df.columns
        assert "longitude" in drone_df.columns
        assert "altitude" in drone_df.columns
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
            lat_col="posllh_lat",
            lon_col="posllh_lon",
            alt_col="posllh_height",
        )
        sync.add_drone_gps(
            sample_drone_blacksquare,
            timestamp_col="timestamp",
            lat_col="Latitude",
            lon_col="Longitude",
            alt_col="heightMSL",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        drone_df = result["drone"]

        # Verify standardized column names
        assert "latitude" in drone_df.columns
        assert "longitude" in drone_df.columns
        assert "altitude" in drone_df.columns
        assert "timestamp" in drone_df.columns

        # Verify original names removed
        assert "Latitude" not in drone_df.columns
        assert "Longitude" not in drone_df.columns
        assert "heightMSL" not in drone_df.columns

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
            lat_col="posllh_lat",
            lon_col="posllh_lon",
            alt_col="posllh_height",
        )
        sync.add_litchi_gps(sample_litchi_gps)

        result = sync.synchronize(target_rate={"drone": 10.0})

        assert "litchi" in result
        litchi_df = result["litchi"]

        # Verify standardized column names (litchi already has lowercase latitude/longitude)
        assert "latitude" in litchi_df.columns
        assert "longitude" in litchi_df.columns
        assert "altitude" in litchi_df.columns
        assert "timestamp" in litchi_df.columns

        # altitude(m) should be renamed to altitude
        assert "altitude(m)" not in litchi_df.columns

    def test_payload_gps_coordinates_standardized(self, sample_gps_payload):
        """Payload GPS outputs standard latitude/longitude/altitude columns."""
        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.add_gps_reference(
            sample_gps_payload,
            timestamp_col="timestamp",
            lat_col="posllh_lat",
            lon_col="posllh_lon",
            alt_col="posllh_height",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        assert "reference_gps" in result
        gps_df = result["reference_gps"]

        # Verify standardized column names
        assert "latitude" in gps_df.columns
        assert "longitude" in gps_df.columns
        assert "altitude" in gps_df.columns
        assert "timestamp" in gps_df.columns

        # Verify original u-blox names removed
        assert "posllh_lat" not in gps_df.columns
        assert "posllh_lon" not in gps_df.columns
        assert "posllh_height" not in gps_df.columns

    def test_all_gps_sources_same_column_names(
        self, sample_gps_payload, sample_drone_dji_rtk, sample_litchi_gps
    ):
        """All GPS sources have identical coordinate column names."""
        from pils.synchronizer import Synchronizer

        sync = Synchronizer()
        sync.add_gps_reference(
            sample_gps_payload,
            timestamp_col="timestamp",
            lat_col="posllh_lat",
            lon_col="posllh_lon",
            alt_col="posllh_height",
        )
        sync.add_drone_gps(
            sample_drone_dji_rtk,
            timestamp_col="correct_timestamp",
            lat_col="RTK:lat_p",
            lon_col="RTK:lon_p",
            alt_col="RTK:hmsl_p",
        )
        sync.add_litchi_gps(sample_litchi_gps)

        result = sync.synchronize(target_rate={"drone": 10.0})

        # Get coordinate columns from each source
        drone_coords = set(result["drone"].columns) & {
            "latitude",
            "longitude",
            "altitude",
            "timestamp",
        }
        litchi_coords = set(result["litchi"].columns) & {
            "latitude",
            "longitude",
            "altitude",
            "timestamp",
        }
        ref_coords = set(result["reference_gps"].columns) & {
            "latitude",
            "longitude",
            "altitude",
            "timestamp",
        }

        # All sources should have the same standard coordinate columns
        assert drone_coords == {"latitude", "longitude", "altitude", "timestamp"}
        assert litchi_coords == {"latitude", "longitude", "altitude", "timestamp"}
        assert ref_coords == {"latitude", "longitude", "altitude", "timestamp"}

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
            lat_col="posllh_lat",
            lon_col="posllh_lon",
            alt_col="posllh_height",
        )
        sync.add_drone_gps(
            sample_drone_dji_rtk,
            timestamp_col="correct_timestamp",
            lat_col="RTK:lat_p",
            lon_col="RTK:lon_p",
            alt_col="RTK:hmsl_p",
        )

        result = sync.synchronize(target_rate={"drone": 10.0})

        drone_df = result["drone"]

        # Values should be reasonable (filter out NaN from interpolation edges)
        # Polars drop_nulls() doesn't remove NaN, so filter explicitly
        valid_lat = drone_df.filter(pl.col("latitude").is_not_nan())["latitude"]
        valid_lon = drone_df.filter(pl.col("longitude").is_not_nan())["longitude"]
        valid_alt = drone_df.filter(pl.col("altitude").is_not_nan())["altitude"]

        assert valid_lat.mean() > 44.5
        assert valid_lat.mean() < 45.5
        assert valid_lon.mean() > 9.5
        assert valid_lon.mean() < 10.5
        assert valid_alt.mean() > 90.0
        assert valid_alt.mean() < 110.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
