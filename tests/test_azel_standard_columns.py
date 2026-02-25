"""
Tests for AZEL Analysis with standardized coordinate column names.

This test suite verifies that the AZEL analysis module correctly uses
the standardized column names (latitude, longitude, altitude, timestamp)
from synchronized data, eliminating the need for conditional column mapping.
"""

import polars as pl
import pytest

from pils.flight import Flight


class TestAZELStandardColumns:
    """Test AZEL analysis with standardized coordinate columns."""

    @pytest.fixture
    def mock_flight_standard_sync(self, tmp_path):
        """Create mock Flight with sync_data using standard column names."""
        # Create campaign structure with metadata directory
        campaign_dir = tmp_path / "campaign"
        metadata_dir = campaign_dir / "metadata"
        metadata_dir.mkdir(parents=True)

        # Create EMLID CSV in metadata directory
        emlid_csv = metadata_dir / "202511_coordinates.csv"
        csv_content = """Name,Longitude,Latitude,Ellipsoidal height
emlid base,-105.0500,40.0500,1050.0
SATP1,-105.0000,40.0000,1000.000
dji rtk base (antenna base),-105.0005,40.0005,1001.000"""
        emlid_csv.write_text(csv_content)

        # Create flight directory
        flight_path = campaign_dir / "flight_standard"
        drone_data_path = flight_path / "drone"
        drone_data_path.mkdir(parents=True)

        flight_info = {"drone_data_folder_path": str(drone_data_path)}
        flight = Flight(flight_info)

        # Create sync_data with STANDARD column names
        flight.sync_data = {
            "drone": pl.DataFrame(
                {
                    "timestamp": [1000.0, 2000.0, 3000.0],
                    "latitude": [40.0001, 40.0002, 40.0003],
                    "longitude": [-105.0001, -105.0002, -105.0003],
                    "altitude": [1000.0, 1001.0, 1002.0],
                    "battery_percent": [85.0, 84.0, 83.0],
                }
            )
        }

        flight._Flight__drone_model = "dji"
        return flight

    def test_sync_data_uses_standard_latitude(self, mock_flight_standard_sync):
        """Test that AZEL analysis uses standard 'latitude' column."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_standard_sync)

        # Verify flight has sync_data with standard columns
        assert "drone" in azel.flight.sync_data
        drone_df = azel.flight.sync_data["drone"]
        assert "latitude" in drone_df.columns
        assert "RTK:lat_p" not in drone_df.columns

        # Run analysis - should use 'latitude' without conditional mapping
        dji_broadcast = {"lat": -22.9597732, "lon": -67.7866847, "alt": 5173.020}
        version = azel.run_analysis(
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        # Verify analysis completed successfully
        assert version is not None
        assert version.azel_data is not None
        assert version.azel_data.height > 0

    def test_sync_data_uses_standard_longitude(self, mock_flight_standard_sync):
        """Test that AZEL analysis uses standard 'longitude' column."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_standard_sync)

        # Verify flight has sync_data with standard columns
        drone_df = azel.flight.sync_data["drone"]
        assert "longitude" in drone_df.columns
        assert "RTK:lon_p" not in drone_df.columns

        # Run analysis - should use 'longitude' without conditional mapping
        dji_broadcast = {"lat": -22.9597732, "lon": -67.7866847, "alt": 5173.020}
        version = azel.run_analysis(
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        assert version is not None

    def test_sync_data_uses_standard_altitude(self, mock_flight_standard_sync):
        """Test that AZEL analysis uses standard 'altitude' column."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_standard_sync)

        # Verify flight has sync_data with standard columns
        drone_df = azel.flight.sync_data["drone"]
        assert "altitude" in drone_df.columns
        assert "RTK:hmsl_p" not in drone_df.columns

        # Run analysis - should use 'altitude' without conditional mapping
        dji_broadcast = {"lat": -22.9597732, "lon": -67.7866847, "alt": 5173.020}
        version = azel.run_analysis(
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        assert version is not None

    def test_sync_data_uses_standard_timestamp(self, mock_flight_standard_sync):
        """Test that AZEL analysis uses standard 'timestamp' column."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_standard_sync)

        # Verify flight has sync_data with standard columns
        drone_df = azel.flight.sync_data["drone"]
        assert "timestamp" in drone_df.columns

        # Run analysis - should use 'timestamp' without conditional mapping
        dji_broadcast = {"lat": -22.9597732, "lon": -67.7866847, "alt": 5173.020}
        version = azel.run_analysis(
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        assert version is not None
        # Verify output has timestamps
        assert "timestamp" in version.azel_data.columns

    def test_no_conditional_column_mapping(self, mock_flight_standard_sync):
        """Test that AZEL no longer uses conditional column name mapping."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_standard_sync)

        # Create sync_data with only standard names (no old names)
        drone_df = azel.flight.sync_data["drone"]

        # Verify NO old column names exist
        old_column_names = [
            "RTK:lat_p",
            "RTK:lon_p",
            "RTK:hmsl_p",
            "timestamp_old",  # old raw timestamp from GPS:dateTimeStamp
            "Latitude",
            "Longitude",
            "heightMSL",
        ]
        for old_name in old_column_names:
            assert old_name not in drone_df.columns, (
                f"{old_name} should not exist in standardized data"
            )

        # Verify analysis works with only standard names
        dji_broadcast = {"lat": -22.9597732, "lon": -67.7866847, "alt": 5173.020}
        version = azel.run_analysis(
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        assert version is not None

    def test_preserve_non_coordinate_columns(self, mock_flight_standard_sync):
        """Test that non-coordinate columns are preserved with original names."""
        from pils.analyze.azel import AZELAnalysis

        azel = AZELAnalysis(mock_flight_standard_sync)

        # Verify non-coordinate column preserved
        drone_df = azel.flight.sync_data["drone"]
        assert "battery_percent" in drone_df.columns

        # Run analysis - non-coordinate columns should not interfere
        dji_broadcast = {"lat": -22.9597732, "lon": -67.7866847, "alt": 5173.020}
        version = azel.run_analysis(
            telescope_name="SATP1",
            dji_broadcast_geod=dji_broadcast,
            drone_timezone_hours=0.0,
        )

        assert version is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
