"""
Tests for PPK Analysis standalone system.

Tests the PPKAnalysis and PPKVersion classes for RTKLIB-based
post-processing of GPS data with smart execution logic.
"""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from pils import store
from pils.analyze.ppk import PPKAnalysis, PPKVersion
from pils.flight import Flight


def write_legacy_ppk_version(
    path: Path,
    version_name: str,
    pos_data: pl.DataFrame,
    stat_data: pl.DataFrame,
    config_hash: str,
    revision_path: Path,
) -> None:
    """A version in the ppk_solution.h5 layout used before pils.store."""
    import json

    import h5py

    from pils.utils.hdf5 import write_table

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        group = f.create_group(version_name)
        write_table(group, "position", pos_data)
        write_table(group, "statistics", stat_data)
        group.attrs["config_hash"] = config_hash
        group.attrs["timestamp"] = "2026-02-01T12:00:00"
        group.attrs["config_params"] = json.dumps({"pos1-posmode": "kinematic"})
        group.attrs["revision_path"] = str(revision_path)


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


class TestPPKAnalysisLoad:
    """Test PPKAnalysis.load (and its alias from_hdf5) with Flight objects."""

    def test_load_with_flight_object(self, mock_flight):
        """Test load accepts Flight object and extracts flight_path."""
        ppk = PPKAnalysis.load(mock_flight)

        assert ppk.flight_path == mock_flight.flight_path
        assert ppk.ppk_dir.exists()
        assert isinstance(ppk.versions, dict)

    def test_load_validates_flight_type(self, tmp_path):
        """Test load raises TypeError for non-Flight objects."""
        flight_path = tmp_path / "flight_005"
        flight_path.mkdir()

        with pytest.raises(TypeError, match="Expected Flight object"):
            PPKAnalysis.load(str(flight_path))

        with pytest.raises(TypeError, match="Expected Flight object"):
            PPKAnalysis.load(flight_path)

    def test_load_validates_flight_path_exists(self, tmp_path):
        """Test load raises ValueError if flight_path is invalid."""
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
            PPKAnalysis.load(flight)


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

    def test_should_run_when_nothing_saved(self, tmp_path):
        """Test that analysis runs when no previous version exists."""
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

        # Simulate a previous run

        # Mock latest version with same config hash
        config_hash = ppk._hash_config(config_file)
        mock_version = PPKVersion(
            version_name="rev_20260204_120000",
            pos_data=pl.DataFrame({"lat_deg": [45.0], "lon_deg": [10.0]}),
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

        # Mock latest version with old config hash
        old_hash = ppk._hash_config(old_config)
        mock_version = PPKVersion(
            version_name="rev_20260204_120000",
            pos_data=pl.DataFrame({"lat_deg": [45.0]}),
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
                "lat_deg": [45.0, 45.001, 45.002],
                "lon_deg": [10.0, 10.001, 10.002],
            }
        )
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "num_sats": [12, 13, 12],
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


class TestSave:
    """Saving PPKVersion (pils.store: ppk/<rev>/)."""

    def test_save_version_creates_version_dir(self, tmp_path):
        """Saving writes the revision's manifest and tables."""
        flight_path = tmp_path / "flight_save"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Create test version
        pos_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "lat_deg": [45.0, 45.001, 45.002],
                "lon_deg": [10.0, 10.001, 10.002],
            }
        )
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "num_sats": [12, 13, 12],
            }
        )
        version = PPKVersion(
            version_name="rev_20260204_143022",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={"config_hash": "abc123", "timestamp": "2026-02-04T14:30:22"},
            revision_path=ppk.ppk_dir / "rev_20260204_143022",
        )

        ppk._save_version(version)

        folder = ppk.ppk_dir / "rev_20260204_143022"
        assert (folder / "manifest.json").is_file()
        # Nothing is written to the legacy HDF5 file
        assert not ppk.hdf5_path.exists()

    def test_save_version_creates_tables(self, tmp_path):
        """The revision has a position and a statistics table."""
        flight_path = tmp_path / "flight_groups"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        pos_df = pl.DataFrame({"timestamp": [1.0], "lat_deg": [45.0]})
        stat_df = pl.DataFrame({"timestamp": [1.0], "num_sats": [12]})
        version = PPKVersion(
            version_name="rev_20260204_143022",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={"config_hash": "test"},
            revision_path=ppk.ppk_dir / "rev_20260204_143022",
        )

        ppk._save_version(version)

        folder = ppk.ppk_dir / "rev_20260204_143022"
        assert (folder / "position.parquet").is_file()
        assert (folder / "statistics.parquet").is_file()
        assert store.list_versions(ppk.ppk_dir) == ["rev_20260204_143022"]

    def test_save_position_dataframe(self, tmp_path):
        """Test that position DataFrame is saved correctly."""
        flight_path = tmp_path / "flight_pos"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        pos_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0, 3.0],
                "lat_deg": [45.0, 45.001, 45.002],
                "lon_deg": [10.0, 10.001, 10.002],
                "h_ell_m": [100.0, 101.0, 102.0],
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

        ppk._save_version(version)

        position = pl.read_parquet(ppk.ppk_dir / "rev_test" / "position.parquet")
        assert position.equals(pos_df)

    def test_save_statistics_dataframe(self, tmp_path):
        """Test that statistics DataFrame is saved correctly."""
        flight_path = tmp_path / "flight_stat"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        pos_df = pl.DataFrame({"timestamp": [1.0]})
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "num_sats": [12, 13],
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

        ppk._save_version(version)

        statistics = pl.read_parquet(ppk.ppk_dir / "rev_test2" / "statistics.parquet")
        assert statistics.equals(stat_df)

    def test_save_metadata_in_manifest(self, tmp_path):
        """Test that metadata is saved in the revision manifest."""
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

        ppk._save_version(version)

        manifest = store.read_manifest(ppk.ppk_dir / "rev_meta_test")
        assert manifest["kind"] == "ppk"
        saved = manifest["metadata"]
        assert saved["config_hash"] == "abc123def456"
        assert saved["timestamp"] == "2026-02-04T14:30:22"
        assert saved["config_params"]["pos1-posmode"] == "kinematic"
        assert saved["revision_path"] == str(ppk.ppk_dir / "rev_meta_test")

    def test_save_multiple_versions(self, tmp_path):
        """Test saving multiple versions."""
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
        ppk._save_version(version1)

        # Save second version
        version2 = PPKVersion(
            version_name="rev_20260204_150000",
            pos_data=pl.DataFrame({"timestamp": [2.0]}),
            stat_data=pl.DataFrame({"timestamp": [2.0]}),
            metadata={"config_hash": "hash2"},
            revision_path=ppk.ppk_dir / "rev_20260204_150000",
        )
        ppk._save_version(version2)

        assert store.list_versions(ppk.ppk_dir) == [
            "rev_20260204_140000",
            "rev_20260204_150000",
        ]

    def test_saved_version_is_not_overwritten(self, tmp_path):
        """A saved revision does not change: saving it again fails."""
        flight_path = tmp_path / "flight_once"
        flight_path.mkdir()
        ppk = PPKAnalysis(create_flight_from_path(flight_path))
        version = PPKVersion(
            version_name="rev_20260204_140000",
            pos_data=pl.DataFrame({"timestamp": [1.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata={},
            revision_path=ppk.ppk_dir / "rev_20260204_140000",
        )
        ppk._save_version(version)
        with pytest.raises(FileExistsError):
            ppk._save_version(version)


class TestLoad:
    """Loading PPKVersion and PPKAnalysis.load."""

    def test_load_version(self, tmp_path):
        """Test loading a saved version."""
        flight_path = tmp_path / "flight_load"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        # Save a version first
        pos_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "lat_deg": [45.0, 45.001],
            }
        )
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "num_sats": [12, 13],
            }
        )
        version = PPKVersion(
            version_name="rev_load_test",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={"config_hash": "loadhash"},
            revision_path=ppk.ppk_dir / "rev_load_test",
        )
        ppk._save_version(version)

        # Load it back
        loaded_version = ppk._load_version("rev_load_test")

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
                "lat_deg": [45.0, 45.001, 45.002],
                "lon_deg": [10.0, 10.001, 10.002],
            }
        )
        stat_df = pl.DataFrame(
            {
                "timestamp": [1.0, 2.0],
                "num_sats": [12, 13],
            }
        )
        version = PPKVersion(
            version_name="rev_reconstruct",
            pos_data=pos_df,
            stat_data=stat_df,
            metadata={},
            revision_path=ppk.ppk_dir / "rev_reconstruct",
        )
        ppk._save_version(version)

        # Load and verify
        loaded = ppk._load_version("rev_reconstruct")

        # Check position data
        assert loaded.pos_data.shape == (3, 3)
        assert "timestamp" in loaded.pos_data.columns
        assert "lat_deg" in loaded.pos_data.columns
        assert loaded.pos_data["lat_deg"].to_list() == [45.0, 45.001, 45.002]

        # Check statistics data
        assert loaded.stat_data.shape == (2, 2)
        assert "num_sats" in loaded.stat_data.columns
        assert loaded.stat_data["num_sats"].to_list() == [12, 13]

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
        ppk._save_version(version)

        # Load and verify metadata
        loaded = ppk._load_version("rev_meta_load")

        assert loaded.metadata["config_hash"] == "metatesthash"
        assert loaded.metadata["timestamp"] == "2026-02-04T15:00:00"
        assert loaded.metadata["config_params"]["pos1-posmode"] == "static"
        assert loaded.metadata["config_params"]["pos1-elmask"] == "10"

    def test_load_classmethod(self, tmp_path):
        """Test loading PPKAnalysis with its saved versions."""
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
        ppk1._save_version(version)

        # Load via classmethod (from_hdf5 is the same)
        ppk2 = PPKAnalysis.load(flight)
        assert PPKAnalysis.from_hdf5(flight).list_versions() == ["rev_classmethod"]

        assert isinstance(ppk2, PPKAnalysis)
        assert ppk2.flight_path == flight_path
        assert len(ppk2.versions) == 1
        assert "rev_classmethod" in ppk2.versions

    def test_load_loads_all_versions(self, tmp_path):
        """Test that load reads all saved versions."""
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
            ppk1._save_version(version)

        # Load all
        ppk2 = PPKAnalysis.load(flight)

        assert len(ppk2.versions) == 3
        assert "rev_20260204140000" in ppk2.versions
        assert "rev_20260204140100" in ppk2.versions
        assert "rev_20260204140200" in ppk2.versions

    def test_load_nothing_saved_returns_empty(self, tmp_path):
        """Test that load returns an empty PPKAnalysis if nothing is saved."""
        flight_path = tmp_path / "flight_nothing_saved"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis.load(flight)

        assert isinstance(ppk, PPKAnalysis)
        assert len(ppk.versions) == 0
        assert ppk.flight_path == flight_path

    def test_load_reads_legacy_hdf5_versions(self, tmp_path):
        """Versions in a ppk_solution.h5 written before the store are loaded."""
        flight_path = tmp_path / "flight_legacy"
        flight_path.mkdir()
        flight = create_flight_from_path(flight_path)
        ppk1 = PPKAnalysis(flight)

        # A store version and a legacy one
        ppk1._save_version(
            PPKVersion(
                version_name="rev_20260301_120000",
                pos_data=pl.DataFrame({"timestamp": [2.0]}),
                stat_data=pl.DataFrame({"timestamp": [2.0]}),
                metadata={"config_hash": "new"},
                revision_path=ppk1.ppk_dir / "rev_20260301_120000",
            )
        )
        # Old names: GPS-time "time", "lat", "Q"; statistics with "tow" only
        write_legacy_ppk_version(
            ppk1.hdf5_path,
            "rev_20260201_120000",
            pl.DataFrame(
                {
                    "time": [datetime(2025, 12, 6, 18, 33, 43)],
                    "lat": [45.0],
                    "height": [5000.0],
                    "Q": [2],
                }
            ),
            pl.DataFrame({"tow": [585223.0], "snr": [40.0]}),
            config_hash="old",
            revision_path=ppk1.ppk_dir / "rev_20260201_120000",
        )

        ppk2 = PPKAnalysis.load(flight)

        assert ppk2.list_versions() == ["rev_20260201_120000", "rev_20260301_120000"]
        legacy = ppk2.get_version("rev_20260201_120000")
        assert legacy is not None
        assert legacy.metadata["config_hash"] == "old"
        assert legacy.metadata["config_params"]["pos1-posmode"] == "kinematic"
        # Renamed to the schema; GPS time (18 leap seconds) to UTC
        pos = legacy.pos_data
        assert pos.columns == ["datetime", "timestamp", "lat_deg", "h_ell_m", "quality"]
        assert pos["datetime"][0] == datetime(2025, 12, 6, 18, 33, 25, tzinfo=UTC)
        assert (
            pos["timestamp"][0]
            == datetime(2025, 12, 6, 18, 33, 25, tzinfo=UTC).timestamp()
        )
        stat = legacy.stat_data
        assert stat["gps_week"].to_list() == [2395]
        assert stat["timestamp"][0] == pytest.approx(pos["timestamp"][0])
        assert stat["snr_dbhz"].to_list() == [40.0]


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
            ppk._save_version(version)
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
            pos_data=pl.DataFrame({"timestamp": [1.0], "lat_deg": [45.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata={"config_hash": "gethash"},
            revision_path=ppk.ppk_dir / "rev_get_test",
        )
        ppk._save_version(version)
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
        ppk._save_version(version)
        ppk.versions["rev_delete_test"] = version

        assert "rev_delete_test" in ppk.versions

        ppk.delete_version("rev_delete_test")

        assert "rev_delete_test" not in ppk.versions

    def test_delete_version_removes_saved_version(self, tmp_path):
        """Test that delete_version removes the saved revision."""
        flight_path = tmp_path / "flight_delete_saved"
        flight_path.mkdir()

        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)

        version = PPKVersion(
            version_name="rev_delete_saved",
            pos_data=pl.DataFrame({"timestamp": [1.0]}),
            stat_data=pl.DataFrame({"timestamp": [1.0]}),
            metadata={},
            revision_path=ppk.ppk_dir / "rev_delete_saved",
        )
        ppk._save_version(version)
        ppk.versions["rev_delete_saved"] = version

        assert store.list_versions(ppk.ppk_dir) == ["rev_delete_saved"]

        ppk.delete_version("rev_delete_saved")

        assert store.list_versions(ppk.ppk_dir) == []
        assert PPKAnalysis.load(flight).list_versions() == []

    def test_delete_version_removes_legacy_hdf5_group(self, tmp_path):
        """A legacy version is removed from ppk_solution.h5 too."""
        import h5py

        flight_path = tmp_path / "flight_delete_legacy"
        flight_path.mkdir()
        flight = create_flight_from_path(flight_path)
        ppk = PPKAnalysis(flight)
        write_legacy_ppk_version(
            ppk.hdf5_path,
            "rev_20260201_120000",
            pl.DataFrame({"timestamp": [1.0]}),
            pl.DataFrame({"timestamp": [1.0]}),
            config_hash="old",
            revision_path=ppk.ppk_dir / "rev_20260201_120000",
        )
        ppk = PPKAnalysis.load(flight)
        assert ppk.list_versions() == ["rev_20260201_120000"]

        ppk.delete_version("rev_20260201_120000")

        with h5py.File(ppk.hdf5_path, "r") as f:
            assert "rev_20260201_120000" not in f
        assert PPKAnalysis.load(flight).list_versions() == []

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
        ppk._save_version(version)
        ppk.versions["rev_delete_folder"] = version

        assert revision_path.exists()

        ppk.delete_version("rev_delete_folder")

        assert not revision_path.exists()


class TestRunAnalysisIntegration:
    """Test run_analysis integration with saving."""

    @patch("pils.analyze.ppk.subprocess.run")
    @patch("pils.analyze.ppk.POSAnalyzer")
    @patch("pils.analyze.ppk.STATAnalyzer")
    def test_run_analysis_saves_version(
        self, mock_stat, mock_pos, mock_subprocess, tmp_path, mock_rtklib_files
    ):
        """Test that run_analysis automatically saves the version."""
        flight_path, config_path, rover_path, base_path, nav_path = mock_rtklib_files

        # Mock subprocess.run to create output files
        def mock_run(*args, **kwargs):
            # Extract command and output file from args
            cmd = args[0] if args else kwargs.get("args", [])
            if cmd and len(cmd) > 2 and cmd[0] == "rnx2rtkp":
                # Find output file argument (-o index + 1)
                try:
                    o_idx = cmd.index("-o")
                    output_file = cmd[o_idx + 1]
                    # Create mock output files
                    Path(output_file).write_text("# Mock position file\n")
                    Path(str(output_file) + ".stat").write_text(
                        "# Mock statistics file\n"
                    )
                except (ValueError, IndexError):
                    pass
            from unittest.mock import MagicMock

            return MagicMock(returncode=0)

        mock_subprocess.side_effect = mock_run

        # Mock analyzers
        mock_pos_instance = MagicMock()
        mock_pos_instance.parse.return_value = pl.DataFrame(
            {"lat_deg": [45.0], "lon_deg": [10.0]}
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

        # The revision is saved, next to the RTKLIB files
        assert store.list_versions(ppk.ppk_dir) == [version.version_name]
        folder = ppk.ppk_dir / version.version_name
        assert (folder / "manifest.json").is_file()
        assert (folder / "position.parquet").is_file()

    @patch("pils.analyze.ppk.subprocess.run")
    @patch("pils.analyze.ppk.POSAnalyzer")
    @patch("pils.analyze.ppk.STATAnalyzer")
    def test_run_analysis_version_loadable(
        self, mock_stat, mock_pos, mock_subprocess, tmp_path, mock_rtklib_files
    ):
        """Test that version created by run_analysis can be loaded."""
        flight_path, config_path, rover_path, base_path, nav_path = mock_rtklib_files

        # Mock subprocess.run to create output files
        def mock_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if cmd and len(cmd) > 2 and cmd[0] == "rnx2rtkp":
                try:
                    o_idx = cmd.index("-o")
                    output_file = cmd[o_idx + 1]
                    Path(output_file).write_text("# Mock position file\n")
                    Path(str(output_file) + ".stat").write_text(
                        "# Mock statistics file\n"
                    )
                except (ValueError, IndexError):
                    pass
            from unittest.mock import MagicMock

            return MagicMock(returncode=0)

        mock_subprocess.side_effect = mock_run

        # Mock analyzers
        mock_pos_instance = MagicMock()
        mock_pos_instance.parse.return_value = pl.DataFrame(
            {"lat_deg": [45.0], "lon_deg": [10.0]}
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

        # Load back
        ppk2 = PPKAnalysis.load(flight)
        version2 = ppk2.get_version(version1.version_name)

        assert version2 is not None
        assert version2.version_name == version1.version_name
        assert version2.metadata["config_hash"] == version1.metadata["config_hash"]

    @patch("pils.analyze.ppk.subprocess.run")
    @patch("pils.analyze.ppk.POSAnalyzer")
    @patch("pils.analyze.ppk.STATAnalyzer")
    def test_run_analysis_multiple_versions_persisted(
        self, mock_stat, mock_pos, mock_subprocess, tmp_path, mock_rtklib_files
    ):
        """Test that multiple run_analysis calls persist all versions."""
        flight_path, config_path, rover_path, base_path, nav_path = mock_rtklib_files

        # Mock subprocess.run to create output files
        def mock_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get("args", [])
            if cmd and len(cmd) > 2 and cmd[0] == "rnx2rtkp":
                try:
                    o_idx = cmd.index("-o")
                    output_file = cmd[o_idx + 1]
                    # Create mock output files
                    Path(output_file).write_text("# Mock position file\n")
                    Path(str(output_file) + ".stat").write_text(
                        "# Mock statistics file\n"
                    )
                except (ValueError, IndexError):
                    pass
            from unittest.mock import MagicMock

            return MagicMock(returncode=0)

        mock_subprocess.side_effect = mock_run

        # Mock analyzers
        mock_pos_instance = MagicMock()
        mock_pos_instance.parse.return_value = pl.DataFrame(
            {"lat_deg": [45.0], "lon_deg": [10.0]}
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
        ppk2 = PPKAnalysis.load(flight)

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
            {"lat_deg": [45.0], "lon_deg": [10.0]}
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
