"""
Tests for PPK Analysis standalone system.

Tests the PPKAnalysis and PPKVersion classes for RTKLIB-based
post-processing of GPS data with smart execution logic.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pils.analyze.ppk import PPKAnalysis, PPKVersion
from pils.flight import Flight


def create_flight_from_path(flight_path: Path) -> Flight:
    """
    Helper function to create a Flight object from a flight_path.

    Used to adapt old tests to new Flight-based API.

    Parameters
    ----------
    flight_path : Path
        Path to flight directory

    Returns
    -------
    Flight
        Flight object with flight_path set
    """
    drone_data_path = flight_path / "drone"
    drone_data_path.mkdir(parents=True, exist_ok=True)
    aux_data_path = flight_path / "aux"
    aux_data_path.mkdir(parents=True, exist_ok=True)

    flight_info = {
        "drone_data_folder_path": str(drone_data_path),
        "aux_data_folder_path": str(aux_data_path),
    }
    return Flight(flight_info)


@pytest.fixture
def mock_flight(tmp_path):
    """
    Create a mock Flight object with valid flight_path.

    Returns:
        Flight instance with flight_path set
    """
    flight_path = tmp_path / "flight_test"
    flight_path.mkdir()
    drone_data_path = flight_path / "drone"
    drone_data_path.mkdir()
    aux_data_path = flight_path / "aux"
    aux_data_path.mkdir()

    flight_info = {
        "drone_data_folder_path": str(drone_data_path),
        "aux_data_folder_path": str(aux_data_path),
    }
    flight = Flight(flight_info)
    return flight


@pytest.fixture
def mock_rtklib_files(tmp_path):
    """
    Create mock RTKLIB input files for testing run_analysis.

    Returns:
        Tuple of (flight_path, config_path, rover_path, base_path, nav_path)
    """
    # Create flight directory
    flight_path = tmp_path / "flight_rtklib"
    flight_path.mkdir()

    # Create mock config file
    config_path = tmp_path / "rtklib.conf"
    config_content = """
# RTKLIB config for testing
pos1-posmode=kinematic
pos1-elmask=15
pos2-armode=fix-and-hold
ant2-postype=llh
"""
    config_path.write_text(config_content)

    # Create mock rover observation file
    rover_path = tmp_path / "rover.obs"
    rover_path.write_text("# Mock RINEX observation file\n")

    # Create mock base observation file
    base_path = tmp_path / "base.obs"
    base_path.write_text("# Mock RINEX observation file\n")

    # Create mock navigation file
    nav_path = tmp_path / "nav.nav"
    nav_path.write_text("# Mock RINEX navigation file\n")

    return flight_path, config_path, rover_path, base_path, nav_path


class TestPPKAnalysisInit:
    """Test PPKAnalysis initialization and path setup."""

    def test_init_with_flight_object(self, mock_flight):
        """Test initialization with Flight object."""
        ppk = PPKAnalysis(mock_flight)

        assert ppk.flight_path == mock_flight.flight_path
        assert ppk.ppk_dir.exists()
        assert ppk.ppk_dir == mock_flight.flight_path / "proc" / "ppk"
        assert ppk.hdf5_path == ppk.ppk_dir / "ppk_solution.h5"

    def test_init_validates_flight_type(self, tmp_path):
        """Test that initialization raises TypeError for non-Flight objects."""
        flight_path = tmp_path / "flight_004"
        flight_path.mkdir()

        with pytest.raises(TypeError, match="Expected Flight object"):
            PPKAnalysis(str(flight_path))

        with pytest.raises(TypeError, match="Expected Flight object"):
            PPKAnalysis(flight_path)

    def test_init_validates_flight_path_exists(self, tmp_path):
        """Test that initialization raises ValueError if flight_path is invalid."""
        # Create flight with None flight_path
        drone_data_path = tmp_path / "drone"
        drone_data_path.mkdir()

        flight_info = {
            "drone_data_folder_path": str(drone_data_path),
            "aux_data_folder_path": str(tmp_path / "aux"),
        }
        flight = Flight(flight_info)

        # Manually set flight_path to None to simulate invalid state
        flight.flight_path = None

        with pytest.raises(
            ValueError, match="Flight object must have a valid flight_path"
        ):
            PPKAnalysis(flight)

    def test_init_validates_flight_path_is_directory(self, tmp_path):
        """Test that initialization raises ValueError if flight_path is not a directory."""
        # Create a file instead of directory
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")

        drone_data_path = tmp_path / "drone"
        drone_data_path.mkdir()

        flight_info = {
            "drone_data_folder_path": str(drone_data_path),
            "aux_data_folder_path": str(tmp_path / "aux"),
        }
        flight = Flight(flight_info)
        flight.flight_path = file_path

        with pytest.raises(
            ValueError, match="flight_path must be an existing directory"
        ):
            PPKAnalysis(flight)


class TestPPKAnalysisFromHDF5:
    """Test PPKAnalysis.from_hdf5 classmethod with Flight objects."""

    def test_from_hdf5_with_flight_object(self, mock_flight):
        """Test from_hdf5 accepts Flight object and extracts flight_path."""
        ppk = PPKAnalysis.from_hdf5(mock_flight)

        assert ppk.flight_path == mock_flight.flight_path
        assert ppk.ppk_dir.exists()
        assert isinstance(ppk.versions, dict)

    def test_from_hdf5_validates_flight_type(self, tmp_path):
        """Test from_hdf5 raises TypeError for non-Flight objects."""
        flight_path = tmp_path / "flight_005"
        flight_path.mkdir()

        with pytest.raises(TypeError, match="Expected Flight object"):
            PPKAnalysis.from_hdf5(str(flight_path))

        with pytest.raises(TypeError, match="Expected Flight object"):
            PPKAnalysis.from_hdf5(flight_path)

    def test_from_hdf5_validates_flight_path_exists(self, tmp_path):
        """Test from_hdf5 raises ValueError if flight_path is invalid."""
        drone_data_path = tmp_path / "drone"
        drone_data_path.mkdir()

        flight_info = {
            "drone_data_folder_path": str(drone_data_path),
            "aux_data_folder_path": str(tmp_path / "aux"),
        }
        flight = Flight(flight_info)
        flight.flight_path = None

        with pytest.raises(
            ValueError, match="Flight object must have a valid flight_path"
        ):
            PPKAnalysis.from_hdf5(flight)


class TestConfigHashing:
    """Test configuration file hashing for change detection."""

    def test_hash_config_returns_sha256(self, tmp_path):
        """Test that config hashing returns valid SHA256 hash."""
        config_file = tmp_path / "test.conf"
        config_file.write_text("pos1-posmode=kinematic\npos1-elmask=15\n")

        flight = create_flight_from_path(tmp_path)
        ppk = PPKAnalysis(flight)
        hash_value = ppk._hash_config(config_file)

        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex length

    def test_hash_config_same_content_same_hash(self, tmp_path):
        """Test that identical config files produce identical hashes."""
        config1 = tmp_path / "config1.conf"
        config2 = tmp_path / "config2.conf"
        content = "pos1-posmode=kinematic\npos1-elmask=15\n"
        config1.write_text(content)
        config2.write_text(content)

        flight = create_flight_from_path(tmp_path)
        ppk = PPKAnalysis(flight)
        hash1 = ppk._hash_config(config1)
        hash2 = ppk._hash_config(config2)

        assert hash1 == hash2

    def test_hash_config_different_content_different_hash(self, tmp_path):
        """Test that different config files produce different hashes."""
        config1 = tmp_path / "config1.conf"
        config2 = tmp_path / "config2.conf"
        config1.write_text("pos1-posmode=kinematic\n")
        config2.write_text("pos1-posmode=static\n")

        flight = create_flight_from_path(tmp_path)
        ppk = PPKAnalysis(flight)
        hash1 = ppk._hash_config(config1)
        hash2 = ppk._hash_config(config2)

        assert hash1 != hash2


class TestConfigParsing:
    """Test RTKLIB config file parameter parsing."""

    def test_parse_config_extracts_key_parameters(self, tmp_path):
        """Test parsing extracts key RTKLIB parameters."""
        config_file = tmp_path / "test.conf"
        config_file.write_text(
            "pos1-posmode=kinematic\n"
            "pos1-elmask=15\n"
            "pos2-armode=fix-and-hold\n"
            "ant2-postype=llh\n"
        )

        flight = create_flight_from_path(tmp_path)
        ppk = PPKAnalysis(flight)
        params = ppk._parse_config_params(config_file)

        assert "pos1-posmode" in params
        assert params["pos1-posmode"] == "kinematic"
        assert "pos1-elmask" in params
        assert params["pos1-elmask"] == "15"
        assert "pos2-armode" in params
        assert params["pos2-armode"] == "fix-and-hold"

    def test_parse_config_ignores_comments(self, tmp_path):
        """Test that comment lines are ignored during parsing."""
        config_file = tmp_path / "test.conf"
        config_file.write_text(
            "# This is a comment\n"
            "pos1-posmode=kinematic\n"
            "# Another comment\n"
            "pos1-elmask=15\n"
        )

        flight = create_flight_from_path(tmp_path)
        ppk = PPKAnalysis(flight)
        params = ppk._parse_config_params(config_file)

        assert "pos1-posmode" in params
        assert "pos1-elmask" in params
        assert len(params) == 2

    def test_parse_config_handles_empty_file(self, tmp_path):
        """Test parsing empty config file returns empty dict."""
        config_file = tmp_path / "empty.conf"
        config_file.write_text("")

        flight = create_flight_from_path(tmp_path)
        ppk = PPKAnalysis(flight)
        params = ppk._parse_config_params(config_file)

        assert isinstance(params, dict)
        assert len(params) == 0


class TestVersionNaming:
    """Test auto-timestamp version name generation."""

    def test_generate_version_name_format(self, tmp_path):
        """Test version name follows rev_YYYYMMDD_HHMMSS format."""
        flight = create_flight_from_path(tmp_path)
        ppk = PPKAnalysis(flight)
        version_name = ppk._generate_version_name()

        assert version_name.startswith("rev_")
        # Format: rev_YYYYMMDD_HHMMSS (total length 19)
        assert len(version_name) == 19

        # Extract parts
        date_part = version_name[4:12]  # YYYYMMDD
        time_part = version_name[13:19]  # HHMMSS

        # Validate date and time parts are numeric
        assert date_part.isdigit()
        assert time_part.isdigit()

    def test_generate_version_name_unique(self, tmp_path):
        """Test that successive calls generate different version names."""
        import time

        flight = create_flight_from_path(tmp_path)
        ppk = PPKAnalysis(flight)
        name1 = ppk._generate_version_name()
        time.sleep(1.1)  # Ensure at least 1 second difference
        name2 = ppk._generate_version_name()

        assert name1 != name2

    def test_generate_version_name_parseable_datetime(self, tmp_path):
        """Test that version name can be parsed back to datetime."""
        flight = create_flight_from_path(tmp_path)
        ppk = PPKAnalysis(flight)
        version_name = ppk._generate_version_name()

        # Extract datetime part (remove 'rev_' prefix)
        dt_string = version_name[4:]

        # Should be parseable
        parsed_dt = datetime.strptime(dt_string, "%Y%m%d_%H%M%S")
        assert isinstance(parsed_dt, datetime)


class TestSmartExecution:
    """Test smart re-run logic for PPK analysis."""

    def test_should_run_when_no_hdf5_exists(self, tmp_path):
        """Test that analysis runs when no previous HDF5 file exists."""
        flight_path = tmp_path / "flight_001"
        flight_path.mkdir()
        config_file = tmp_path / "test.conf"
        config_file.write_text("pos1-posmode=kinematic\n")

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        assert ppk._should_run_analysis(config_file) is True

    def test_should_not_run_when_config_unchanged(self, tmp_path):
        """Test that analysis skips when config hasn't changed."""
        flight_path = tmp_path / "flight_002"
        flight_path.mkdir()
        config_file = tmp_path / "test.conf"
        config_file.write_text("pos1-posmode=kinematic\n")

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Create mock HDF5 file and simulate previous run
        ppk.hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        ppk.hdf5_path.touch()

        # Mock latest version with same config hash
        config_hash = ppk._hash_config(config_file)
        mock_version = PPKVersion(
            version_name="rev_20260204_120000",
            pos_data=pl.DataFrame({"lat": [45.0], "lon": [10.0]}),
            stat_data=pl.DataFrame({"sat": [12]}),
            metadata={"config_hash": config_hash},
            revision_path=ppk.ppk_dir / "rev_20260204_120000",
        )
        ppk.versions = {"rev_20260204_120000": mock_version}

        assert ppk._should_run_analysis(config_file) is False

    def test_should_run_when_config_changed(self, tmp_path):
        """Test that analysis runs when config has changed."""
        flight_path = tmp_path / "flight_003"
        flight_path.mkdir()
        old_config = tmp_path / "old.conf"
        new_config = tmp_path / "new.conf"
        old_config.write_text("pos1-posmode=kinematic\n")
        new_config.write_text("pos1-posmode=static\n")  # Changed

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)
        ppk.hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        ppk.hdf5_path.touch()

        # Mock latest version with old config hash
        old_hash = ppk._hash_config(old_config)
        mock_version = PPKVersion(
            version_name="rev_20260204_120000",
            pos_data=pl.DataFrame({"lat": [45.0]}),
            stat_data=pl.DataFrame({"sat": [12]}),
            metadata={"config_hash": old_hash},
            revision_path=ppk.ppk_dir / "rev_20260204_120000",
        )
        ppk.versions = {"rev_20260204_120000": mock_version}

        # Check with new config
        assert ppk._should_run_analysis(new_config) is True


class TestRevisionFolderCreation:
    """Test creation of per-revision folders."""

    def test_create_revision_folder(self, tmp_path):
        """Test that revision folder is created correctly."""
        flight_path = tmp_path / "flight_001"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)
        version_name = "rev_20260204_143022"

        revision_path = ppk._create_revision_folder(version_name)

        assert revision_path.exists()
        assert revision_path.is_dir()
        assert revision_path == ppk.ppk_dir / version_name

    def test_create_revision_folder_preserves_existing(self, tmp_path):
        """Test that existing revision folder is not deleted."""
        flight_path = tmp_path / "flight_002"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)
        version_name = "rev_20260204_143022"

        # Create folder with marker file
        ppk.ppk_dir.mkdir(parents=True, exist_ok=True)
        existing_folder = ppk.ppk_dir / version_name
        existing_folder.mkdir()
        marker = existing_folder / "marker.txt"
        marker.write_text("existing")

        revision_path = ppk._create_revision_folder(version_name)

        assert revision_path == existing_folder
        assert marker.exists()
        assert marker.read_text() == "existing"


class TestPPKVersionDataclass:
    """Test PPKVersion dataclass structure."""

    def test_ppk_version_creation(self):
        """Test creating PPKVersion instance."""
        pos_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "lat": [45.0, 45.001, 45.002],
                "lon": [10.0, 10.001, 10.002],
            }
        )
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "num_sat": [12, 13, 12],
            }
        )
        metadata = {
            "config_hash": "abc123",
            "timestamp": "2026-02-04T14:30:22",
        }

        version = PPKVersion(
            version_name="rev_20260204_143022",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata=metadata,
            revision_path=Path("/flight/proc/ppk/rev_20260204_143022"),
        )

        assert version.version_name == "rev_20260204_143022"
        assert isinstance(version.pos_data, pl.DataFrame)
        assert isinstance(version.stat_data, pl.DataFrame)
        assert version.metadata["config_hash"] == "abc123"
        assert isinstance(version.revision_path, Path)

    def test_ppk_version_attributes(self):
        """Test that PPKVersion has all required attributes."""
        version = PPKVersion(
            version_name="test",
            pos_data=pl.DataFrame(),
            stat_data=pl.DataFrame(),
            metadata={},
            revision_path=Path("."),
        )

        assert hasattr(version, "version_name")
        assert hasattr(version, "pos_data")
        assert hasattr(version, "stat_data")
        assert hasattr(version, "metadata")
        assert hasattr(version, "revision_path")


class TestHDF5Save:
    """Test HDF5 save operations for PPKVersion."""

    def test_save_version_creates_hdf5_file(self, tmp_path):
        """Test that saving version creates HDF5 file."""
        flight_path = tmp_path / "flight_hdf5"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Create test version
        pos_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "lat": [45.0, 45.001, 45.002],
                "lon": [10.0, 10.001, 10.002],
            }
        )
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "num_sat": [12, 13, 12],
            }
        )
        version = PPKVersion(
            version_name="rev_20260204_143022",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={"config_hash": "abc123", "timestamp": "2026-02-04T14:30:22"},
            revision_path=ppk.ppk_dir / "rev_20260204_143022",
        )

        # Save to HDF5
        ppk._save_version_to_hdf5(version)

        # Verify file exists
        assert ppk.hdf5_path.exists()

    def test_save_version_creates_version_group(self, tmp_path):
        """Test that saving creates correct version group structure."""
        import h5py

        flight_path = tmp_path / "flight_groups"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        pos_df = pl.DataFrame({"timestamp": [1.0], "lat": [45.0]})
        stat_df = pl.DataFrame({"timestamp": [1.0], "num_sat": [12]})
        version = PPKVersion(
            version_name="rev_20260204_143022",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={"config_hash": "test"},
            revision_path=ppk.ppk_dir / "rev_20260204_143022",
        )

        ppk._save_version_to_hdf5(version)

        # Check HDF5 structure
        with h5py.File(ppk.hdf5_path, "r") as f:
            assert "rev_20260204_143022" in f
            version_group = f["rev_20260204_143022"]
            assert "position" in version_group
            assert "statistics" in version_group

    def test_save_position_dataframe(self, tmp_path):
        """Test that position DataFrame is saved correctly."""
        import h5py

        flight_path = tmp_path / "flight_pos"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        pos_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "lat": [45.0, 45.001, 45.002],
                "lon": [10.0, 10.001, 10.002],
                "height": [100.0, 101.0, 102.0],
            }
        )
        stat_df = pl.DataFrame({"timestamp": [1.0]})
        version = PPKVersion(
            version_name="rev_test",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={},
            revision_path=ppk.ppk_dir / "rev_test",
        )

        ppk._save_version_to_hdf5(version)

        # Verify position data structure
        with h5py.File(ppk.hdf5_path, "r") as f:
            pos_group = f["rev_test/position"]
            assert "timestamp" in pos_group
            assert "lat" in pos_group
            assert "lon" in pos_group
            assert "height" in pos_group
            assert len(pos_group["timestamp"]) == 3

    def test_save_statistics_dataframe(self, tmp_path):
        """Test that statistics DataFrame is saved correctly."""
        import h5py

        flight_path = tmp_path / "flight_stat"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        pos_df = pl.DataFrame({"timestamp": [1.0]})
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "num_sat": [12, 13],
                "ratio": [2.5, 3.0],
            }
        )
        version = PPKVersion(
            version_name="rev_test2",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={},
            revision_path=ppk.ppk_dir / "rev_test2",
        )

        ppk._save_version_to_hdf5(version)

        # Verify statistics data
        with h5py.File(ppk.hdf5_path, "r") as f:
            stat_group = f["rev_test2/statistics"]
            assert "timestamp" in stat_group
            assert "num_sat" in stat_group
            assert "ratio" in stat_group
            assert len(stat_group["num_sat"]) == 2

    def test_save_metadata_attrs(self, tmp_path):
        """Test that metadata is saved as group attributes."""
        import json

        import h5py

        flight_path = tmp_path / "flight_meta"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        pos_df = pl.DataFrame({"timestamp": [1.0]})
        stat_df = pl.DataFrame({"timestamp": [1.0]})
        metadata = {
            "config_hash": "abc123def456",
            "timestamp": "2026-02-04T14:30:22",
            "config_params": {"pos1-posmode": "kinematic", "pos1-elmask": "15"},
        }
        version = PPKVersion(
            version_name="rev_meta_test",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata=metadata,
            revision_path=ppk.ppk_dir / "rev_meta_test",
        )

        ppk._save_version_to_hdf5(version)

        # Verify metadata attributes
        with h5py.File(ppk.hdf5_path, "r") as f:
            version_group = f["rev_meta_test"]
            assert "config_hash" in version_group.attrs
            assert "timestamp" in version_group.attrs
            assert "config_params" in version_group.attrs
            assert "revision_path" in version_group.attrs

            # Check values
            assert version_group.attrs["config_hash"] == "abc123def456"
            assert version_group.attrs["timestamp"] == "2026-02-04T14:30:22"

            # Config params should be JSON string
            config_params = json.loads(version_group.attrs["config_params"])
            assert config_params["pos1-posmode"] == "kinematic"

    def test_save_multiple_versions(self, tmp_path):
        """Test saving multiple versions to same HDF5 file."""
        import h5py

        flight_path = tmp_path / "flight_multi"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Save first version
        version1 = PPKVersion(
            version_name="rev_20260204_140000",
            pos_data=pl.DataFrame({"timestamp": [1.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata={"config_hash": "hash1"},
            revision_path=ppk.ppk_dir / "rev_20260204_140000",
        )
        ppk._save_version_to_hdf5(version1)

        # Save second version
        version2 = PPKVersion(
            version_name="rev_20260204_150000",
            pos_data=pl.DataFrame({"timestamp": [2.0]}),
            stat_data=pl.DataFrame({"timestamp": [2.0]}),
            metadata={"config_hash": "hash2"},
            revision_path=ppk.ppk_dir / "rev_20260204_150000",
        )
        ppk._save_version_to_hdf5(version2)

        # Verify both versions exist
        with h5py.File(ppk.hdf5_path, "r") as f:
            assert "rev_20260204_140000" in f
            assert "rev_20260204_150000" in f


class TestHDF5Load:
    """Test HDF5 load operations for PPKVersion."""

    def test_load_version_from_hdf5(self, tmp_path):
        """Test loading a version from HDF5."""
        flight_path = tmp_path / "flight_load"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Save a version first
        pos_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "lat": [45.0, 45.001],
            }
        )
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "num_sat": [12, 13],
            }
        )
        version = PPKVersion(
            version_name="rev_load_test",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={"config_hash": "loadhash"},
            revision_path=ppk.ppk_dir / "rev_load_test",
        )
        ppk._save_version_to_hdf5(version)

        # Load it back
        loaded_version = ppk._load_version_from_hdf5("rev_load_test")

        assert loaded_version.version_name == "rev_load_test"
        assert isinstance(loaded_version.pos_data, pl.DataFrame)
        assert isinstance(loaded_version.stat_data, pl.DataFrame)

    def test_load_version_reconstructs_dataframes(self, tmp_path):
        """Test that loaded version has correct DataFrame data."""
        flight_path = tmp_path / "flight_reconstruct"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Original data
        pos_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "lat": [45.0, 45.001, 45.002],
                "lon": [10.0, 10.001, 10.002],
            }
        )
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "num_sat": [12, 13],
            }
        )
        version = PPKVersion(
            version_name="rev_reconstruct",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={},
            revision_path=ppk.ppk_dir / "rev_reconstruct",
        )
        ppk._save_version_to_hdf5(version)

        # Load and verify
        loaded = ppk._load_version_from_hdf5("rev_reconstruct")

        # Check position data
        assert loaded.pos_data.shape == (3, 3)
        assert "timestamp" in loaded.pos_data.columns
        assert "lat" in loaded.pos_data.columns
        assert loaded.pos_data["lat"].to_list() == [45.0, 45.001, 45.002]

        # Check statistics data
        assert loaded.stat_data.shape == (2, 2)
        assert "num_sat" in loaded.stat_data.columns
        assert loaded.stat_data["num_sat"].to_list() == [12, 13]

    def test_load_version_reconstructs_metadata(self, tmp_path):
        """Test that loaded version has correct metadata."""
        flight_path = tmp_path / "flight_meta_load"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        metadata = {
            "config_hash": "metatesthash",
            "timestamp": "2026-02-04T15:00:00",
            "config_params": {"pos1-posmode": "static", "pos1-elmask": "10"},
        }
        version = PPKVersion(
            version_name="rev_meta_load",
            pos_data=pl.DataFrame({"timestamp": [1.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata=metadata,
            revision_path=ppk.ppk_dir / "rev_meta_load",
        )
        ppk._save_version_to_hdf5(version)

        # Load and verify metadata
        loaded = ppk._load_version_from_hdf5("rev_meta_load")

        assert loaded.metadata["config_hash"] == "metatesthash"
        assert loaded.metadata["timestamp"] == "2026-02-04T15:00:00"
        assert loaded.metadata["config_params"]["pos1-posmode"] == "static"
        assert loaded.metadata["config_params"]["pos1-elmask"] == "10"

    def test_from_hdf5_classmethod(self, tmp_path):
        """Test loading PPKAnalysis from existing HDF5 file."""
        flight_path = tmp_path / "flight_classmethod"
        flight_path.mkdir()

        # Create and save versions
        flight = create_flight_from_path(flight_path)
        ppk1 = PPKAnalysis(flight)
        version = PPKVersion(
            version_name="rev_classmethod",
            pos_data=pl.DataFrame({"timestamp": [1.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata={"config_hash": "classhash"},
            revision_path=ppk1.ppk_dir / "rev_classmethod",
        )
        ppk1._save_version_to_hdf5(version)

        # Load via classmethod
        ppk2 = PPKAnalysis.from_hdf5(flight)

        assert isinstance(ppk2, PPKAnalysis)
        assert ppk2.flight_path == flight_path
        assert len(ppk2.versions) == 1
        assert "rev_classmethod" in ppk2.versions

    def test_from_hdf5_loads_all_versions(self, tmp_path):
        """Test that from_hdf5 loads all stored versions."""
        flight_path = tmp_path / "flight_all_versions"
        flight_path.mkdir()

        # Save multiple versions
        flight = create_flight_from_path(flight_path)
        ppk1 = PPKAnalysis(flight)
        for i in range(3):
            version = PPKVersion(
                version_name=f"rev_2026020414{i:02d}00",
                pos_data=pl.DataFrame({"timestamp": [float(i)]}),
                stat_data=pl.DataFrame({"timestamp": [float(i)]}),
                metadata={"config_hash": f"hash{i}"},
                revision_path=ppk1.ppk_dir / f"rev_2026020414{i:02d}00",
            )
            ppk1._save_version_to_hdf5(version)

        # Load all
        ppk2 = PPKAnalysis.from_hdf5(flight)

        assert len(ppk2.versions) == 3
        assert "rev_20260204140000" in ppk2.versions
        assert "rev_20260204140100" in ppk2.versions
        assert "rev_20260204140200" in ppk2.versions

    def test_from_hdf5_no_file_returns_empty(self, tmp_path):
        """Test that from_hdf5 returns empty PPKAnalysis if no HDF5 file exists."""
        flight_path = tmp_path / "flight_no_hdf5"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis.from_hdf5(flight)

        assert isinstance(ppk, PPKAnalysis)
        assert len(ppk.versions) == 0
        assert ppk.flight_path == flight_path


class TestVersionManagement:
    """Test version management operations."""

    def test_list_versions_sorted(self, tmp_path):
        """Test that list_versions returns sorted version names."""
        flight_path = tmp_path / "flight_list"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Add versions in random order
        for name in [
            "rev_20260204_150000",
            "rev_20260204_140000",
            "rev_20260204_160000",
        ]:
            version = PPKVersion(
                version_name=name,
                pos_data=pl.DataFrame({"timestamp": [1.0]}),
                stat_data=pl.DataFrame({"timestamp": [1.0]}),
                metadata={},
                revision_path=ppk.ppk_dir / name,
            )
            ppk._save_version_to_hdf5(version)
            ppk.versions[name] = version

        versions = ppk.list_versions()

        # Should be sorted chronologically
        assert versions == [
            "rev_20260204_140000",
            "rev_20260204_150000",
            "rev_20260204_160000",
        ]

    def test_get_version_by_name(self, tmp_path):
        """Test retrieving specific version by name."""
        flight_path = tmp_path / "flight_get"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        version = PPKVersion(
            version_name="rev_get_test",
            pos_data=pl.DataFrame({"timestamp": [1.0], "lat": [45.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata={"config_hash": "gethash"},
            revision_path=ppk.ppk_dir / "rev_get_test",
        )
        ppk._save_version_to_hdf5(version)
        ppk.versions["rev_get_test"] = version

        retrieved = ppk.get_version("rev_get_test")

        assert retrieved is not None
        assert retrieved.version_name == "rev_get_test"
        assert retrieved.metadata["config_hash"] == "gethash"

    def test_get_version_nonexistent_returns_none(self, tmp_path):
        """Test that get_version returns None for nonexistent version."""
        flight_path = tmp_path / "flight_nonexistent"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        result = ppk.get_version("rev_does_not_exist")

        assert result is None

    def test_delete_version_removes_from_dict(self, tmp_path):
        """Test that delete_version removes from versions dict."""
        flight_path = tmp_path / "flight_delete_dict"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        version = PPKVersion(
            version_name="rev_delete_test",
            pos_data=pl.DataFrame({"timestamp": [1.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata={},
            revision_path=ppk.ppk_dir / "rev_delete_test",
        )
        ppk._save_version_to_hdf5(version)
        ppk.versions["rev_delete_test"] = version

        assert "rev_delete_test" in ppk.versions

        ppk.delete_version("rev_delete_test")

        assert "rev_delete_test" not in ppk.versions

    def test_delete_version_removes_from_hdf5(self, tmp_path):
        """Test that delete_version removes group from HDF5 file."""
        import h5py

        flight_path = tmp_path / "flight_delete_hdf5"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        version = PPKVersion(
            version_name="rev_delete_hdf5",
            pos_data=pl.DataFrame({"timestamp": [1.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata={},
            revision_path=ppk.ppk_dir / "rev_delete_hdf5",
        )
        ppk._save_version_to_hdf5(version)
        ppk.versions["rev_delete_hdf5"] = version

        # Verify exists in HDF5
        with h5py.File(ppk.hdf5_path, "r") as f:
            assert "rev_delete_hdf5" in f

        ppk.delete_version("rev_delete_hdf5")

        # Verify removed from HDF5
        with h5py.File(ppk.hdf5_path, "r") as f:
            assert "rev_delete_hdf5" not in f

    def test_delete_version_removes_folder(self, tmp_path):
        """Test that delete_version removes revision folder from filesystem."""
        flight_path = tmp_path / "flight_delete_folder"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Create revision folder
        revision_path = ppk.ppk_dir / "rev_delete_folder"
        revision_path.mkdir()
        (revision_path / "solution.pos").write_text("test")

        version = PPKVersion(
            version_name="rev_delete_folder",
            pos_data=pl.DataFrame({"timestamp": [1.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata={},
            revision_path=revision_path,
        )
        ppk._save_version_to_hdf5(version)
        ppk.versions["rev_delete_folder"] = version

        assert revision_path.exists()

        ppk.delete_version("rev_delete_folder")

        assert not revision_path.exists()


class TestRunAnalysisIntegration:
    """Test run_analysis integration with HDF5 save."""

    @patch("pils.analyze.ppk.RTKLIBRunner")
    @patch("pils.analyze.ppk.POSAnalyzer")
    @patch("pils.analyze.ppk.STATAnalyzer")
    def test_run_analysis_saves_to_hdf5(
        self, mock_stat, mock_pos, mock_rtklib, tmp_path, mock_rtklib_files
    ):
        """Test that run_analysis automatically saves to HDF5."""
        flight_path, config_path, rover_path, base_path, nav_path = mock_rtklib_files

        # Mock RTKLIB runner
        mock_rtklib_instance = MagicMock()
        mock_rtklib_instance.check_overlap.return_value = True

        def mock_run_ppk(rover, base, nav, config, output):
            # Create mock output files that RTKLIB would create
            Path(output).write_text("# Mock position file\n")
            Path(str(output) + ".stat").write_text("# Mock statistics file\n")

        mock_rtklib_instance.run_ppk.side_effect = mock_run_ppk
        mock_rtklib.return_value = mock_rtklib_instance

        # Mock analyzers
        mock_pos_instance = MagicMock()
        mock_pos_instance.parse.return_value = pl.DataFrame(
            {"lat": [45.0], "lon": [10.0]}
        )
        mock_pos.return_value = mock_pos_instance

        mock_stat_instance = MagicMock()
        mock_stat_instance.parse.return_value = pl.DataFrame({"sat": [12]})
        mock_stat.return_value = mock_stat_instance

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        version = ppk.run_analysis(
            config_path, rover_path, base_path, nav_path, force=True
        )

        # HDF5 file should exist
        assert ppk.hdf5_path.exists()

        # Version should be in HDF5
        import h5py

        with h5py.File(ppk.hdf5_path, "r") as f:
            assert version.version_name in f

    @patch("pils.analyze.ppk.RTKLIBRunner")
    @patch("pils.analyze.ppk.POSAnalyzer")
    @patch("pils.analyze.ppk.STATAnalyzer")
    def test_run_analysis_version_loadable(
        self, mock_stat, mock_pos, mock_rtklib, tmp_path, mock_rtklib_files
    ):
        """Test that version created by run_analysis can be loaded."""
        flight_path, config_path, rover_path, base_path, nav_path = mock_rtklib_files

        # Mock RTKLIB runner
        mock_rtklib_instance = MagicMock()
        mock_rtklib_instance.check_overlap.return_value = True

        def mock_run_ppk(rover, base, nav, config, output):
            # Create mock output files that RTKLIB would create
            Path(output).write_text("# Mock position file\n")
            Path(str(output) + ".stat").write_text("# Mock statistics file\n")

        mock_rtklib_instance.run_ppk.side_effect = mock_run_ppk
        mock_rtklib.return_value = mock_rtklib_instance

        # Mock analyzers
        mock_pos_instance = MagicMock()
        mock_pos_instance.parse.return_value = pl.DataFrame(
            {"lat": [45.0], "lon": [10.0]}
        )
        mock_pos.return_value = mock_pos_instance

        mock_stat_instance = MagicMock()
        mock_stat_instance.parse.return_value = pl.DataFrame({"sat": [12]})
        mock_stat.return_value = mock_stat_instance

        flight = create_flight_from_path(flight_path)
        ppk1 = PPKAnalysis(flight)
        version1 = ppk1.run_analysis(
            config_path, rover_path, base_path, nav_path, force=True
        )

        # Load from HDF5
        ppk2 = PPKAnalysis.from_hdf5(flight)
        version2 = ppk2.get_version(version1.version_name)

        assert version2 is not None
        assert version2.version_name == version1.version_name
        assert version2.metadata["config_hash"] == version1.metadata["config_hash"]

    @patch("pils.analyze.ppk.RTKLIBRunner")
    @patch("pils.analyze.ppk.POSAnalyzer")
    @patch("pils.analyze.ppk.STATAnalyzer")
    def test_run_analysis_multiple_versions_persisted(
        self, mock_stat, mock_pos, mock_rtklib, tmp_path, mock_rtklib_files
    ):
        """Test that multiple run_analysis calls persist all versions."""
        flight_path, config_path, rover_path, base_path, nav_path = mock_rtklib_files

        # Mock RTKLIB runner
        mock_rtklib_instance = MagicMock()
        mock_rtklib_instance.check_overlap.return_value = True

        def mock_run_ppk(rover, base, nav, config, output):
            # Create mock output files that RTKLIB would create
            Path(output).write_text("# Mock position file\n")
            Path(str(output) + ".stat").write_text("# Mock statistics file\n")

        mock_rtklib_instance.run_ppk.side_effect = mock_run_ppk
        mock_rtklib.return_value = mock_rtklib_instance

        # Mock analyzers
        mock_pos_instance = MagicMock()
        mock_pos_instance.parse.return_value = pl.DataFrame(
            {"lat": [45.0], "lon": [10.0]}
        )
        mock_pos.return_value = mock_pos_instance

        mock_stat_instance = MagicMock()
        mock_stat_instance.parse.return_value = pl.DataFrame({"sat": [12]})
        mock_stat.return_value = mock_stat_instance

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Run analysis twice with force
        version1 = ppk.run_analysis(
            config_path, rover_path, base_path, nav_path, force=True
        )

        # Sleep to ensure different timestamp
        import time

        time.sleep(1.1)

        version2 = ppk.run_analysis(
            config_path, rover_path, base_path, nav_path, force=True
        )

        # Both should be different and persisted
        assert version1.version_name != version2.version_name

        # Load fresh instance
        ppk2 = PPKAnalysis.from_hdf5(flight)

        assert len(ppk2.versions) == 2
        assert version1.version_name in ppk2.versions
        assert version2.version_name in ppk2.versions


class TestRevisionFolderCleanup:
    """Test cleanup of revision folders when PPK analysis fails."""

    @patch("subprocess.run")
    def test_cleanup_folder_when_pos_file_not_created(self, mock_subprocess, tmp_path):
        """Test that revision folder is removed when pos file is not created."""
        flight_path = tmp_path / "flight_cleanup_test"
        flight_path.mkdir()

        # Create config and input files
        config_path = tmp_path / "rtklib.conf"
        config_path.write_text("pos1-posmode=kinematic\n")
        rover_path = tmp_path / "rover.obs"
        rover_path.write_text("# Mock rover\n")
        base_path = tmp_path / "base.obs"
        base_path.write_text("# Mock base\n")
        nav_path = tmp_path / "nav.nav"
        nav_path.write_text("# Mock nav\n")

        # Mock subprocess to not create output files (simulating failure)
        mock_subprocess.return_value = MagicMock(returncode=0)

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Run analysis - should fail because pos file not created
        result = ppk.run_analysis(
            config_path, rover_path, base_path, nav_path, force=True
        )

        # Should return None
        assert result is None

        # Revision folder should not exist
        revision_folders = list(ppk.ppk_dir.glob("rev_*"))
        assert len(revision_folders) == 0

    @patch("subprocess.run")
    @patch("pils.analyze.ppk.POSAnalyzer")
    def test_cleanup_folder_when_parsing_fails(
        self, mock_pos_analyzer, mock_subprocess, tmp_path
    ):
        """Test that revision folder is removed when position file parsing fails."""
        flight_path = tmp_path / "flight_parse_fail"
        flight_path.mkdir()

        # Create config and input files
        config_path = tmp_path / "rtklib.conf"
        config_path.write_text("pos1-posmode=kinematic\n")
        rover_path = tmp_path / "rover.obs"
        rover_path.write_text("# Mock rover\n")
        base_path = tmp_path / "base.obs"
        base_path.write_text("# Mock base\n")
        nav_path = tmp_path / "nav.nav"
        nav_path.write_text("# Mock nav\n")

        # Mock subprocess to create output files
        def create_pos_file(*args, **kwargs):
            # Find the -o argument to get output path
            if "-o" in args[0]:
                output_idx = args[0].index("-o") + 1
                pos_file = Path(args[0][output_idx])
                pos_file.write_text("# Mock pos file\n")
            return MagicMock(returncode=0)

        mock_subprocess.side_effect = create_pos_file

        # Mock POSAnalyzer to raise exception
        mock_pos_analyzer.return_value.parse.side_effect = ValueError("Parse error")

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Run analysis - should fail during parsing
        result = ppk.run_analysis(
            config_path, rover_path, base_path, nav_path, force=True
        )

        # Should return None
        assert result is None

        # Revision folder should not exist
        revision_folders = list(ppk.ppk_dir.glob("rev_*"))
        assert len(revision_folders) == 0

    @patch("subprocess.run")
    @patch("pils.analyze.ppk.POSAnalyzer")
    @patch("pils.analyze.ppk.STATAnalyzer")
    def test_preserves_folder_on_success(
        self, mock_stat, mock_pos, mock_subprocess, tmp_path
    ):
        """Test that revision folder is preserved when analysis succeeds."""
        flight_path = tmp_path / "flight_success"
        flight_path.mkdir()

        # Create config and input files
        config_path = tmp_path / "rtklib.conf"
        config_path.write_text("pos1-posmode=kinematic\n")
        rover_path = tmp_path / "rover.obs"
        rover_path.write_text("# Mock rover\n")
        base_path = tmp_path / "base.obs"
        base_path.write_text("# Mock base\n")
        nav_path = tmp_path / "nav.nav"
        nav_path.write_text("# Mock nav\n")

        # Mock subprocess to create output files
        def create_output_files(*args, **kwargs):
            # Find the -o argument to get output path
            if "-o" in args[0]:
                output_idx = args[0].index("-o") + 1
                pos_file = Path(args[0][output_idx])
                pos_file.write_text("# Mock pos file\n")
                stat_file = Path(str(pos_file) + ".stat")
                stat_file.write_text("# Mock stat file\n")
            return MagicMock(returncode=0)

        mock_subprocess.side_effect = create_output_files

        # Mock analyzers to succeed
        mock_pos.return_value.parse.return_value = pl.DataFrame(
            {"lat": [45.0], "lon": [10.0]}
        )
        mock_stat.return_value.parse.return_value = pl.DataFrame({"sat": [12]})

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Run analysis - should succeed
        result = ppk.run_analysis(
            config_path, rover_path, base_path, nav_path, force=True
        )

        # Should return PPKVersion
        assert result is not None
        assert isinstance(result, PPKVersion)

        # Revision folder should exist
        assert result.revision_path.exists()
        assert result.revision_path.is_dir()
