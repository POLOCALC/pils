"""Tests for Flight.sync() method and sync_data persistence (parquet)."""

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
    """sync() raises without a payload GPS and without a drone GPS."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    with pytest.raises(ValueError, match="No reference timebase"):
        flight.sync()


def test_sync_data_persists_as_parquet(sample_flight_with_gps, tmp_path):
    """sync_data is saved by Flight.save and loaded back by Flight.load."""
    flight = sample_flight_with_gps
    flight.sync(target_rate={"drone": 1.0})

    revision = flight.save(tmp_path / "flight")
    assert (tmp_path / "flight" / "sync" / revision / "manifest.json").is_file()

    loaded = Flight.load(tmp_path / "flight")
    assert isinstance(loaded.sync_data, dict)
    assert loaded.sync_data.keys() == flight.sync_data.keys()
    for key, table in flight.sync_data.items():
        assert isinstance(loaded.sync_data[key], pl.DataFrame)
        assert loaded.sync_data[key].shape == table.shape
        assert loaded.sync_data[key].columns == table.columns


def test_a_flight_without_sync_data_saves_no_sync_revision(tmp_path):
    """Without sync_data only the raw data is saved."""
    flight = Flight(
        {
            "drone_data_folder_path": "/tmp/test_flight/drone",
            "aux_data_folder_path": "/tmp/test_flight/aux",
        }
    )
    assert flight.save(tmp_path / "flight") is None
    assert Flight.load(tmp_path / "flight").sync_data is None


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

    flight.save(tmp_path / "flight")
    loaded = Flight.load(tmp_path / "flight")
    assert (
        loaded.sync_data["drone_imu_atti_0"].shape == synced["drone_imu_atti_0"].shape
    )
