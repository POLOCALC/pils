"""Test HDF5 structure for sync_data with dictionary format."""

import tempfile
import time
from pathlib import Path

import h5py
import polars as pl

from pils.flight import Flight


def test_sync_data_dict_structure_saves_to_hdf5():
    """Test that sync_data dictionary is saved with correct HDF5 structure."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    # Manually create sync_data as a dictionary (simulating synchronizer output)
    flight.sync_data = {
        "drone": pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "lat": [45.0, 45.001, 45.002],
                "lon": [10.0, 10.001, 10.002],
            }
        ),
        "reference_gps": pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "posllh_lat": [45.0, 45.001, 45.002],
                "posllh_lon": [10.0, 10.001, 10.002],
                "posllh_height": [100.0, 101.0, 102.0],
            }
        ),
        "payload": pl.DataFrame(
            {
                "adc_voltage": [1.1, 1.2, 1.3],
                "adc_current": [0.5, 0.6, 0.7],
            }
        ),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_structure.h5"

        # Save to HDF5
        flight.to_hdf5(hdf5_path)

        # Verify HDF5 structure
        with h5py.File(hdf5_path, "r") as f:
            # Check sync_data group exists
            assert "sync_data" in f
            sync_group = f["sync_data"]

            # Check revision group exists
            revision_keys = [k for k in sync_group.keys() if k.startswith("rev_")]
            assert len(revision_keys) == 1

            revision_group = sync_group[revision_keys[0]]

            # Check all keys from sync_data dict are present
            assert "drone" in revision_group
            assert "reference_gps" in revision_group
            assert "payload" in revision_group

            # Check metadata attributes
            assert "created_at" in revision_group.attrs
            assert "n_keys" in revision_group.attrs
            assert "pils_version" in revision_group.attrs
            assert revision_group.attrs["n_keys"] == 3

            # Check each dataset has correct structure
            drone_group = revision_group["drone"]
            assert "timestamp" in drone_group
            assert "lat" in drone_group
            assert "lon" in drone_group
            assert "columns" in drone_group.attrs
            assert "dtypes" in drone_group.attrs


def test_sync_data_dict_loads_from_hdf5():
    """Test that sync_data dictionary is loaded correctly from HDF5."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    # Manually create sync_data as a dictionary
    original_sync_data = {
        "drone": pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "lat": [45.0, 45.001, 45.002],
                "lon": [10.0, 10.001, 10.002],
            }
        ),
        "reference_gps": pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "posllh_lat": [45.0, 45.001, 45.002],
                "posllh_lon": [10.0, 10.001, 10.002],
                "posllh_height": [100.0, 101.0, 102.0],
            }
        ),
    }
    flight.sync_data = original_sync_data

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_structure.h5"

        # Save to HDF5
        flight.to_hdf5(hdf5_path)

        # Load from HDF5
        loaded_flight = Flight.from_hdf5(hdf5_path)

        # Verify sync_data was loaded as dictionary
        assert loaded_flight.sync_data is not None
        assert isinstance(loaded_flight.sync_data, dict)

        # Verify all keys are present
        assert set(loaded_flight.sync_data.keys()) == set(original_sync_data.keys())

        # Verify each DataFrame matches
        for key in original_sync_data.keys():
            assert key in loaded_flight.sync_data
            assert isinstance(loaded_flight.sync_data[key], pl.DataFrame)
            assert loaded_flight.sync_data[key].shape == original_sync_data[key].shape
            assert (
                loaded_flight.sync_data[key].columns == original_sync_data[key].columns
            )


def test_sync_version_parameter_works():
    """Test that sync_version parameter correctly loads specific revisions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_versions.h5"

        # Manually create HDF5 file with multiple revisions
        with h5py.File(hdf5_path, "w") as f:
            # Create metadata group
            metadata_group = f.create_group("metadata")
            metadata_group.attrs["flight_info_drone_data_folder_path"] = (
                "/tmp/test_flight/drone"
            )
            metadata_group.attrs["flight_info_aux_data_folder_path"] = (
                "/tmp/test_flight/aux"
            )

            # Create sync_data with multiple revisions
            sync_data_group = f.create_group("sync_data")

            # First revision
            rev1_group = sync_data_group.create_group("rev_20260201_120000")
            drone1_group = rev1_group.create_group("drone")
            drone1_group.create_dataset("timestamp", data=[0.0, 1.0])
            drone1_group.create_dataset("version", data=[1.0, 1.0])
            drone1_group.attrs["columns"] = '["timestamp", "version"]'
            drone1_group.attrs["dtypes"] = '["Float64", "Float64"]'
            drone1_group.attrs["n_rows"] = 2
            rev1_group.attrs["created_at"] = "rev_20260201_120000"
            rev1_group.attrs["n_keys"] = 1

            # Second revision (later)
            rev2_group = sync_data_group.create_group("rev_20260202_150000")
            drone2_group = rev2_group.create_group("drone")
            drone2_group.create_dataset("timestamp", data=[0.0, 1.0, 2.0])
            drone2_group.create_dataset("version", data=[2.0, 2.0, 2.0])
            drone2_group.attrs["columns"] = '["timestamp", "version"]'
            drone2_group.attrs["dtypes"] = '["Float64", "Float64"]'
            drone2_group.attrs["n_rows"] = 3
            rev2_group.attrs["created_at"] = "rev_20260202_150000"
            rev2_group.attrs["n_keys"] = 1

        # Load without specifying version (should load latest)
        flight_latest = Flight.from_hdf5(hdf5_path)
        assert flight_latest.sync_data is not None
        assert "drone" in flight_latest.sync_data
        assert flight_latest.sync_data["drone"].shape == (3, 2)
        assert flight_latest.sync_data["drone"]["version"][0] == 2.0

        # Load specific older version
        flight_v1 = Flight.from_hdf5(hdf5_path, sync_version="rev_20260201_120000")
        assert flight_v1.sync_data is not None
        assert "drone" in flight_v1.sync_data
        assert flight_v1.sync_data["drone"].shape == (2, 2)
        assert flight_v1.sync_data["drone"]["version"][0] == 1.0

        # Load specific newer version explicitly
        flight_v2 = Flight.from_hdf5(hdf5_path, sync_version="rev_20260202_150000")
        assert flight_v2.sync_data is not None
        assert "drone" in flight_v2.sync_data
        assert flight_v2.sync_data["drone"].shape == (3, 2)
        assert flight_v2.sync_data["drone"]["version"][0] == 2.0

        # Load with sync_version=False (should not load any sync data)
        flight_no_sync = Flight.from_hdf5(hdf5_path, sync_version=False)
        assert flight_no_sync.sync_data is None


def test_sync_metadata_saves_to_hdf5():
    """Test that user-provided sync_metadata is saved to HDF5."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    # Create sync_data dictionary
    flight.sync_data = {
        "drone": pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "lat": [45.0, 45.001, 45.002],
                "lon": [10.0, 10.001, 10.002],
            }
        ),
        "reference_gps": pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "posllh_lat": [45.0, 45.001, 45.002],
                "posllh_lon": [10.0, 10.001, 10.002],
            }
        ),
    }

    # Define user metadata
    sync_metadata = {
        "comment": "Initial synchronization test",
        "target_rate": 10.0,
        "processing_version": "v1.2.3",
        "notes": "Test run with high accuracy GPS",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_metadata.h5"

        # Save with sync metadata
        flight.to_hdf5(hdf5_path, sync_metadata=sync_metadata)

        # Verify metadata was saved in HDF5 file
        with h5py.File(hdf5_path, "r") as f:
            assert "sync_data" in f
            sync_group = f["sync_data"]

            # Get the revision group
            revision_keys = [k for k in sync_group.keys() if k.startswith("rev_")]
            assert len(revision_keys) == 1
            revision_group = sync_group[revision_keys[0]]

            # Check standard metadata
            assert "created_at" in revision_group.attrs
            assert "n_keys" in revision_group.attrs
            assert "pils_version" in revision_group.attrs

            # Check user metadata (prefixed with "user_")
            assert "user_comment" in revision_group.attrs
            assert (
                revision_group.attrs["user_comment"] == "Initial synchronization test"
            )

            assert "user_target_rate" in revision_group.attrs
            assert revision_group.attrs["user_target_rate"] == 10.0

            assert "user_processing_version" in revision_group.attrs
            assert revision_group.attrs["user_processing_version"] == "v1.2.3"

            assert "user_notes" in revision_group.attrs
            assert (
                revision_group.attrs["user_notes"] == "Test run with high accuracy GPS"
            )


def test_multiple_revisions_with_different_metadata():
    """Test that multiple revisions can have different metadata."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    # Create sync_data dictionary
    flight.sync_data = {
        "drone": pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "lat": [45.0, 45.001, 45.002],
            }
        ),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_multi_metadata.h5"

        # Save first revision with metadata
        flight.to_hdf5(hdf5_path, sync_metadata={"comment": "First sync", "rate": 10.0})

        # Wait to ensure different timestamp
        time.sleep(1.1)

        # Modify data and save second revision with different metadata
        flight.sync_data["drone"] = pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0, 3.0],
                "lat": [45.0, 45.001, 45.002, 45.003],
            }
        )
        flight.to_hdf5(
            hdf5_path, sync_metadata={"comment": "Second sync", "rate": 20.0}
        )

        # Verify both revisions have different metadata
        with h5py.File(hdf5_path, "r") as f:
            sync_group = f["sync_data"]
            revision_keys = sorted(
                [k for k in sync_group.keys() if k.startswith("rev_")]
            )
            assert len(revision_keys) == 2

            # Check first revision
            rev1_group = sync_group[revision_keys[0]]
            assert rev1_group.attrs["user_comment"] == "First sync"
            assert rev1_group.attrs["user_rate"] == 10.0

            # Check second revision
            rev2_group = sync_group[revision_keys[1]]
            assert rev2_group.attrs["user_comment"] == "Second sync"
            assert rev2_group.attrs["user_rate"] == 20.0


def test_sync_without_metadata_still_works():
    """Test that saving without sync_metadata still works (None is default)."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    flight.sync_data = {
        "drone": pl.DataFrame(
            {
                "timestamp": [0.0, 1.0, 2.0],
                "lat": [45.0, 45.001, 45.002],
            }
        ),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        hdf5_path = Path(tmpdir) / "test_no_metadata.h5"

        # Save without sync_metadata
        flight.to_hdf5(hdf5_path)

        # Verify standard metadata exists but no user metadata
        with h5py.File(hdf5_path, "r") as f:
            sync_group = f["sync_data"]
            revision_keys = [k for k in sync_group.keys() if k.startswith("rev_")]
            revision_group = sync_group[revision_keys[0]]

            # Standard metadata should exist
            assert "created_at" in revision_group.attrs
            assert "n_keys" in revision_group.attrs
            assert "pils_version" in revision_group.attrs

            # No user metadata should exist
            user_attrs = [
                k for k in revision_group.attrs.keys() if k.startswith("user_")
            ]
            assert len(user_attrs) == 0
