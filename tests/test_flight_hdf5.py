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

    def test_save_raw_data_creates_payload_data_group(
        self, sample_flight, temp_h5_file
    ):
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

    def test_save_raw_data_preserves_dataframe_structure(
        self, sample_flight, temp_h5_file
    ):
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


class TestColumnTypesRoundTrip:
    """Column types found in real flights survive to_hdf5/from_hdf5."""

    def test_strings_datetimes_and_table_groups(self, temp_h5_file):
        flight = Flight({"drone_data_folder_path": "/tmp/test_flight/drone"})
        drone = pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "datetime": ["2025-12-08 19:27:00", None, "2025-12-08 19:27:02"],
            }
        )
        flight.raw_data.drone_data = DroneData(drone_df=drone, litchi_df=None)
        times = pl.Series(
            [datetime(2025, 12, 8, 19, 27), datetime(2025, 12, 8, 19, 28)]
        )
        flight.raw_data.payload_data.gps = pl.DataFrame(
            {
                "timestamp": [0.0, 1.0],
                "utc": times,
                "utc_tz": times.dt.replace_time_zone("UTC").dt.cast_time_unit("ns"),
            }
        )
        # e.g. the inclinometer: one table per message type
        flight.raw_data.payload_data.inclinometer = {
            "INS": pl.DataFrame({"timestamp": [0.0], "roll": [1.0]}),
            "IMU": pl.DataFrame({"timestamp": [0.0, 0.5], "acc_x": [0.1, 0.2]}),
        }

        flight.to_hdf5(temp_h5_file)
        loaded = Flight.from_hdf5(temp_h5_file)

        loaded_drone = loaded.raw_data.drone_data.drone
        assert loaded_drone["datetime"].to_list() == [
            "2025-12-08 19:27:00",
            None,  # nulls are stored as a mask
            "2025-12-08 19:27:02",
        ]
        gps = loaded.raw_data.payload_data.gps
        assert gps.equals(flight.raw_data.payload_data.gps)
        assert gps.schema == flight.raw_data.payload_data.gps.schema
        # Tables and columns come back with the schema's names
        inclinometer = loaded.raw_data.payload_data.inclinometer
        assert set(inclinometer) == {"ins", "imu"}
        assert inclinometer["ins"].columns == ["timestamp", "roll_deg"]
        assert inclinometer["imu"].equals(
            flight.raw_data.payload_data.inclinometer["IMU"]
        )

    def test_columns_are_compressed(self, temp_h5_file):
        flight = Flight({"drone_data_folder_path": "/tmp/test_flight/drone"})
        flight.raw_data.payload_data.gps = pl.DataFrame(
            {"timestamp": np.arange(1000.0)}
        )
        # Empty arrays cannot be chunked, so they are stored uncompressed
        flight.raw_data.payload_data.lm76 = pl.DataFrame(
            {"timestamp": pl.Series([], dtype=pl.Float64)}
        )
        flight.to_hdf5(temp_h5_file)
        with h5py.File(temp_h5_file, "r") as f:
            assert f["raw_data/payload_data/gps/timestamp"].compression == "gzip"
            assert f["raw_data/payload_data/lm76/timestamp"].shape == (0,)

    def test_dat_records_round_trip(self, temp_h5_file):
        flight = Flight({"drone_data_folder_path": "/tmp/test_flight/drone"})
        records = {
            "IMU_ATTI(0)": pl.DataFrame(
                {
                    "IMU_ATTI(0):roll": [1.0, 2.0],
                    "tick": [10, 20],
                    "timestamp": [0.0, 1.0],
                }
            ),
            "Motor": pl.DataFrame({"Motor:Speed:RFront": [3800], "tick": [15]}),
        }
        flight.raw_data.drone_data = DroneData(
            drone_df=pl.DataFrame({"timestamp": [0.0]}), litchi_df=None, records=records
        )
        flight.to_hdf5(temp_h5_file)

        # Older files: records and columns get the schema's names
        loaded = Flight.from_hdf5(temp_h5_file).raw_data.drone_data.records
        assert set(loaded) == {"imu_atti_0", "motor"}
        assert loaded["imu_atti_0"].columns == ["roll_deg", "tick", "timestamp"]
        assert loaded["imu_atti_0"]["roll_deg"].to_list() == [1.0, 2.0]
        assert loaded["motor"].columns == ["speed_rfront_rpm", "tick"]


class TestTableFormat:
    """Nulls, units and types through pils.utils.hdf5.write_table/read_table."""

    def test_nulls_types_and_units(self, temp_h5_file):
        import h5py

        from pils.utils.hdf5 import read_table, read_units, write_table

        frame = pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "rtk_lat_deg": [1.0, None, float("nan")],
                "numSV": pl.Series([3, None, 5], dtype=pl.Int64),
                "flag": [True, None, False],
                "name": ["a", None, "c"],
                "when": pl.Series(
                    [datetime(2025, 1, 1), None, datetime(2025, 1, 2)]
                ).dt.replace_time_zone("UTC"),
            }
        )
        with h5py.File(temp_h5_file, "w") as f:
            write_table(f, "t", frame)
            loaded = read_table(f["t"])
            units = read_units(f["t"])
        assert loaded.schema == frame.schema
        assert loaded["rtk_lat_deg"].is_null().to_list() == [False, True, False]
        assert np.isnan(loaded["rtk_lat_deg"][2])  # a NaN stays a NaN
        assert loaded.select(pl.all().exclude("rtk_lat_deg")).equals(
            frame.select(pl.all().exclude("rtk_lat_deg"))
        )
        assert units == {"timestamp": "s", "rtk_lat_deg": "deg"}

    def test_legacy_ppk_microseconds(self, temp_h5_file):
        import json

        import h5py

        from pils.utils.hdf5 import read_table

        with h5py.File(temp_h5_file, "w") as f:
            group = f.create_group("t")
            group.create_dataset("t", data=np.array([1_000_000]))
            group.attrs["columns"] = json.dumps(["t"])
            group.attrs["dtypes"] = json.dumps(
                ["Datetime(time_unit='ns', time_zone=None)"]
            )
            loaded = read_table(group, legacy_time_unit="us")
        assert loaded["t"][0] == datetime(1970, 1, 1, 0, 0, 1)
        assert loaded.schema["t"] == pl.Datetime("ns")
