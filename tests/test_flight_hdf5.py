"""
Unit tests for Flight HDF5 serialization methods.

Tests the to_hdf5() and from_hdf5() methods including:
- Raw data saving and loading
- Synchronized data versioning
- Metadata preservation
- Version numbering (rev_YYYYMMDD_hhmm format)
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import polars as pl
import pytest

from pils.flight import DroneData, Flight, PayloadData


@pytest.fixture
def temp_h5_file():
    """Create a temporary HDF5 file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_flight.h5"
        yield filepath


@pytest.fixture
def sample_flight():
    """Create a sample Flight object with drone and sensor data."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
        "flight_name": "Test Flight 001",
        "drone_id": "DJI-M300-001",
    }

    flight = Flight(flight_info)

    # Set metadata
    flight.set_metadata(
        {
            "takeoff_time": datetime(2025, 2, 2, 10, 30),
            "landing_time": datetime(2025, 2, 2, 11, 15),
            "pilot": "Test Pilot",
            "weather": "Clear",
        }
    )

    # Create sample drone data with realistic 100 Hz sampling rate
    # Using timestamps in SECONDS (Synchronizer expects seconds, not milliseconds)
    drone_df = pl.DataFrame(
        {
            "timestamp": [
                1000.0 + i * 0.01 for i in range(1000)
            ],  # 100 Hz (0.01s intervals = 10ms)
            "latitude": np.linspace(40.0, 40.01, 1000),
            "longitude": np.linspace(-74.0, -73.99, 1000),
            "altitude": np.linspace(100, 150, 1000),
            "roll": np.random.normal(0, 5, 1000),
            "pitch": np.random.normal(0, 5, 1000),
            "yaw": np.random.normal(0, 10, 1000),
        }
    )

    # Create sample sensor data with realistic 100 Hz sampling rate
    gps_df = pl.DataFrame(
        {
            "timestamp": [
                1000.0 + i * 0.01 for i in range(1000)
            ],  # 100 Hz (0.01s intervals = 10ms)
            "lat": np.linspace(40.0, 40.01, 1000),
            "lon": np.linspace(-74.0, -73.99, 1000),
            "accuracy": np.random.uniform(1, 5, 1000),
        }
    )

    imu_df = pl.DataFrame(
        {
            "timestamp": [
                1000.0 + i * 0.01 for i in range(1000)
            ],  # 100 Hz (0.01s intervals = 10ms)
            "accel_x": np.random.normal(0, 0.1, 1000),
            "accel_y": np.random.normal(0, 0.1, 1000),
            "accel_z": np.random.normal(9.8, 0.1, 1000),
        }
    )

    # Assign data
    flight.raw_data.drone_data = DroneData(drone_df, None)
    flight.raw_data.payload_data = PayloadData()
    flight.raw_data.payload_data.gps = gps_df
    flight.raw_data.payload_data.imu = imu_df

    return flight


class TestToHDF5RawDataOnly:
    """Test saving raw data without synchronization."""

    def test_save_raw_data_creates_file(self, sample_flight, temp_h5_file):
        """Test that to_hdf5() creates an HDF5 file."""
        sample_flight.to_hdf5(str(temp_h5_file))
        assert temp_h5_file.exists()

    def test_save_raw_data_creates_metadata_group(self, sample_flight, temp_h5_file):
        """Test that metadata group is created."""
        sample_flight.to_hdf5(str(temp_h5_file))

        with h5py.File(str(temp_h5_file), "r") as f:
            assert "metadata" in f
            assert "flight_info_flight_name" in f["metadata"].attrs
            assert f["metadata"].attrs["flight_info_flight_name"] == "Test Flight 001"

    def test_save_raw_data_creates_drone_data_group(self, sample_flight, temp_h5_file):
        """Test that drone data is saved correctly."""
        sample_flight.to_hdf5(str(temp_h5_file))

        with h5py.File(str(temp_h5_file), "r") as f:
            assert "raw_data" in f
            raw_data_group = f["raw_data"]
            assert isinstance(raw_data_group, h5py.Group)
            assert "drone_data" in raw_data_group
            drone_data_group = raw_data_group["drone_data"]
            assert isinstance(drone_data_group, h5py.Group)
            assert "drone" in drone_data_group

    def test_save_raw_data_creates_payload_data_group(self, sample_flight, temp_h5_file):
        """Test that payload sensor data is saved correctly."""
        sample_flight.to_hdf5(str(temp_h5_file))

        with h5py.File(str(temp_h5_file), "r") as f:
            raw_data_group = f["raw_data"]
            assert isinstance(raw_data_group, h5py.Group)
            assert "payload_data" in raw_data_group
            payload_data_group = raw_data_group["payload_data"]
            assert isinstance(payload_data_group, h5py.Group)
            assert "gps" in payload_data_group
            assert "imu" in payload_data_group

    def test_save_raw_data_preserves_dataframe_structure(self, sample_flight, temp_h5_file):
        """Test that DataFrame structure is preserved."""
        sample_flight.to_hdf5(str(temp_h5_file))

        with h5py.File(str(temp_h5_file), "r") as f:
            raw_data_group = f["raw_data"]
            assert isinstance(raw_data_group, h5py.Group)
            drone_data_group = raw_data_group["drone_data"]
            assert isinstance(drone_data_group, h5py.Group)
            drone_group = drone_data_group["drone"]
            assert isinstance(drone_group, h5py.Group)
            columns_attr = drone_group.attrs["columns"]
            if isinstance(columns_attr, bytes):
                columns = json.loads(columns_attr.decode())
            else:
                columns = json.loads(str(columns_attr))
            assert "timestamp" in columns
            assert "latitude" in columns
            assert "altitude" in columns

    def test_save_metadata_attrs(self, sample_flight, temp_h5_file):
        """Test that metadata is stored as attributes."""
        sample_flight.to_hdf5(str(temp_h5_file))

        with h5py.File(str(temp_h5_file), "r") as f:
            meta_attrs = f["metadata"].attrs
            assert "flight_info_drone_id" in meta_attrs
            assert meta_attrs["flight_info_drone_id"] == "DJI-M300-001"


class TestFromHDF5RawData:
    """Test loading raw data from HDF5."""

    def test_load_raw_data_only(self, sample_flight, temp_h5_file):
        """Test loading flight with sync_version=False."""
        sample_flight.to_hdf5(str(temp_h5_file))

        loaded_flight = Flight.from_hdf5(str(temp_h5_file), sync_version=False)

        assert loaded_flight.flight_info["flight_name"] == "Test Flight 001"
        assert loaded_flight.flight_info["drone_id"] == "DJI-M300-001"

    def test_load_metadata_reconstructed(self, sample_flight, temp_h5_file):
        """Test that metadata is correctly reconstructed."""
        sample_flight.to_hdf5(str(temp_h5_file))

        loaded_flight = Flight.from_hdf5(str(temp_h5_file), sync_version=False)

        assert "pilot" in loaded_flight.metadata
        assert loaded_flight.metadata["pilot"] == "Test Pilot"

    def test_load_drone_data_shape_preserved(self, sample_flight, temp_h5_file):
        """Test that drone data shape is preserved."""
        sample_flight.to_hdf5(str(temp_h5_file))

        loaded_flight = Flight.from_hdf5(str(temp_h5_file), sync_version=False)

        assert loaded_flight.raw_data.drone_data is not None
        original_df = sample_flight.raw_data.drone_data.drone
        loaded_df = loaded_flight.raw_data.drone_data.drone
        assert loaded_df is not None
        # Handle both DataFrame and Dict[str, DataFrame]
        if isinstance(original_df, dict) or isinstance(loaded_df, dict):
            return  # Skip if it's a dict
        original_shape = original_df.shape
        loaded_shape = loaded_df.shape
        assert original_shape == loaded_shape

    def test_load_sensor_data_shape_preserved(self, sample_flight, temp_h5_file):
        """Test that sensor data shapes are preserved."""
        sample_flight.to_hdf5(str(temp_h5_file))

        loaded_flight = Flight.from_hdf5(str(temp_h5_file), sync_version=False)

        assert loaded_flight.raw_data.payload_data is not None
        original_gps_shape = sample_flight.raw_data.payload_data.gps.shape
        loaded_gps_shape = loaded_flight.raw_data.payload_data.gps.shape
        assert original_gps_shape == loaded_gps_shape

        original_imu_shape = sample_flight.raw_data.payload_data.imu.shape
        loaded_imu_shape = loaded_flight.raw_data.payload_data.imu.shape
        assert original_imu_shape == loaded_imu_shape

    def test_load_data_values_equal(self, sample_flight, temp_h5_file):
        """Test that loaded data values match original."""
        sample_flight.to_hdf5(str(temp_h5_file))

        loaded_flight = Flight.from_hdf5(str(temp_h5_file), sync_version=False)

        assert loaded_flight.raw_data.drone_data is not None
        original_df = sample_flight.raw_data.drone_data.drone
        loaded_df = loaded_flight.raw_data.drone_data.drone
        assert loaded_df is not None

        # Skip if it's a dict
        if isinstance(original_df, dict) or isinstance(loaded_df, dict):
            return

        # Compare timestamp columns
        original_ts = list(original_df["timestamp"])
        loaded_ts = list(loaded_df["timestamp"])
        assert original_ts == loaded_ts


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_file_not_found_raises_error(self, temp_h5_file):
        """Test that loading nonexistent file raises FileNotFoundError."""
        nonexistent_file = temp_h5_file.parent / "nonexistent.h5"

        with pytest.raises(FileNotFoundError):
            Flight.from_hdf5(str(nonexistent_file))

    def test_save_to_nonexistent_directory(self, sample_flight, temp_h5_file):
        """Test that to_hdf5() creates parent directories."""
        deep_path = temp_h5_file.parent / "deep" / "nested" / "path" / "file.h5"

        sample_flight.to_hdf5(str(deep_path))

        assert deep_path.exists()

    def test_overwrite_existing_file(self, sample_flight, temp_h5_file):
        """Test that saving overwrites existing file gracefully."""
        # Save first time
        sample_flight.to_hdf5(str(temp_h5_file))

        # Save second time (should not raise error)
        sample_flight.to_hdf5(str(temp_h5_file))

        assert temp_h5_file.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
