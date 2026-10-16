"""Tests for Flight.sync() method and sync_data HDF5 persistence."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from pils.flight import Flight


@pytest.fixture
def sample_flight_with_gps():
    """Create a flight with sample GPS data."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    # Add sample GPS payload data
    gps_data = pl.DataFrame(
        {
            "timestamp": [0.0, 1.0, 2.0, 3.0, 4.0],
            "lat_deg": [45.0, 45.001, 45.002, 45.003, 45.004],
            "lon_deg": [10.0, 10.001, 10.002, 10.003, 10.004],
            "h_ell_m": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    # Initialize payload data structure
    from pils.flight import PayloadData

    flight.raw_data.payload_data = PayloadData()
    flight.raw_data.payload_data.gps = gps_data

    return flight


def test_sync_creates_sync_data(sample_flight_with_gps):
    """Test that sync() creates sync_data attribute."""
    flight = sample_flight_with_gps

    # Initially sync_data should be None
    assert flight.sync_data is None

    # Call sync
    result = flight.sync(target_rate={"drone": 1.0})

    # Check sync_data is created as a dictionary
    assert flight.sync_data is not None
    assert isinstance(flight.sync_data, dict)
    assert result is flight.sync_data

    # Check that dictionary values are DataFrames
    for key, value in flight.sync_data.items():
        assert isinstance(value, pl.DataFrame), f"Expected DataFrame for key {key}"


def test_sync_requires_gps_payload():
    """Test that sync() raises error without GPS payload."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    with pytest.raises(ValueError, match="GPS payload data is required"):
        flight.sync()


def test_sync_data_persists_to_hdf5(sample_flight_with_gps):
    """Test that sync_data is saved and loaded from HDF5."""
    flight = sample_flight_with_gps

    # Perform synchronization
    flight.sync(target_rate={"drone": 1.0})

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_flight.h5"

        # Save to HDF5
        flight.to_hdf5(hdf5_path)

        # Load from HDF5
        loaded_flight = Flight.from_hdf5(hdf5_path)

        # Check sync_data was loaded as a dictionary
        assert loaded_flight.sync_data is not None
        assert isinstance(loaded_flight.sync_data, dict)

        # Verify all keys are present
        assert loaded_flight.sync_data.keys() == flight.sync_data.keys()

        # Verify each DataFrame matches
        for key in flight.sync_data.keys():
            assert key in loaded_flight.sync_data
            assert isinstance(loaded_flight.sync_data[key], pl.DataFrame)
            assert loaded_flight.sync_data[key].shape == flight.sync_data[key].shape
            assert loaded_flight.sync_data[key].columns == flight.sync_data[key].columns


def test_sync_data_not_saved_if_none():
    """Test that sync_data is not saved if it doesn't exist."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_flight.h5"

        # Save to HDF5 without sync_data
        flight.to_hdf5(hdf5_path)

        # Load from HDF5
        loaded_flight = Flight.from_hdf5(hdf5_path)

        # sync_data should still be None
        assert loaded_flight.sync_data is None


def test_sync_drone_records(tmp_path):
    """Selected DJI .dat records are synchronized and saved as drone_<name>."""
    import numpy as np

    from pils.flight import DroneData, PayloadData

    flight = Flight({"drone_data_folder_path": "", "aux_data_folder_path": ""})
    t = np.linspace(0, 100, 1000)
    position = {
        "lat": 45.0 + 0.001 * np.sin(0.1 * t),
        "lon": 10.0 + 0.001 * np.cos(0.1 * t),
        "h": 100.0 + 10.0 * np.sin(0.05 * t),
    }
    flight.raw_data.payload_data = PayloadData()
    flight.raw_data.payload_data.gps = pl.DataFrame(
        {
            "timestamp": t,
            "lat_deg": position["lat"],
            "lon_deg": position["lon"],
            "h_ell_m": position["h"],
        }
    )
    drone = pl.DataFrame(
        {
            "timestamp": t + 0.3,
            "rtk_lat_deg": position["lat"],
            "rtk_lon_deg": position["lon"],
            "rtk_h_ell_m": position["h"],
        }
    )
    imu = pl.DataFrame({"yaw_deg": t % 360 - 180, "timestamp": t + 0.3})
    motor = pl.DataFrame({"speed_rfront_rpm": t, "timestamp": t + 0.3})
    flight.raw_data.drone_data = DroneData(
        drone, pl.DataFrame(), {"imu_atti_0": imu, "motor": motor}
    )
    flight._Flight__drone_model = "dji"

    synced = flight.sync(target_rate={"drone": 10.0})
    assert "drone_imu_atti_0" in synced
    assert "drone_motor" not in synced
    assert flight.sync_offsets["drone_gps"] == pytest.approx(-0.3, abs=0.05)

    synced = flight.sync(target_rate={"drone": 10.0}, drone_records="all")
    assert {"drone_imu_atti_0", "drone_motor"} <= set(synced)

    path = tmp_path / "flight.h5"
    flight.to_hdf5(path)
    loaded = Flight.from_hdf5(path)
    assert (
        loaded.sync_data["drone_imu_atti_0"].shape == synced["drone_imu_atti_0"].shape
    )
