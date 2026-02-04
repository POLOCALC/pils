"""Tests for PathLoader."""

from datetime import datetime
from pathlib import Path

import pytest

from pils.loader.path import PathLoader


@pytest.fixture
def mock_campaign_structure(tmp_path):
    """Create mock campaign directory structure."""
    # Create structure: base/campaigns/202511/20251208/flight_20251208_1506/
    base = tmp_path / "data"
    campaign = base / "campaigns" / "202511" / "20251208"
    flight = campaign / "flight_20251208_1506"

    # Create directories
    flight.mkdir(parents=True, exist_ok=True)
    drone_dir = flight / "drone"
    aux_dir = flight / "aux"
    proc_dir = flight / "proc"
    drone_dir.mkdir()
    aux_dir.mkdir()
    proc_dir.mkdir()

    # Create mock sensor files in aux
    (aux_dir / "gps.csv").write_text("timestamp,lat,lon\n1000,45.0,10.0\n")
    (aux_dir / "imu.csv").write_text("timestamp,ax,ay,az\n1000,0,0,9.8\n")

    # Create a second flight
    flight2 = campaign / "flight_20251208_1510"
    flight2.mkdir()
    (flight2 / "drone").mkdir()
    (flight2 / "aux").mkdir()
    (flight2 / "proc").mkdir()

    return base, flight


@pytest.fixture
def mock_multi_campaign_structure(tmp_path):
    """Create mock structure with multiple campaigns."""
    base = tmp_path / "data"
    campaigns_dir = base / "campaigns"

    # Campaign 1
    camp1 = campaigns_dir / "202511" / "20251208"
    flight1 = camp1 / "flight_20251208_1506"
    flight1.mkdir(parents=True)
    (flight1 / "drone").mkdir()
    (flight1 / "aux").mkdir()
    (flight1 / "proc").mkdir()

    # Campaign 2
    camp2 = campaigns_dir / "202512" / "20251215"
    flight2 = camp2 / "flight_20251215_1400"
    flight2.mkdir(parents=True)
    (flight2 / "drone").mkdir()
    (flight2 / "aux").mkdir()
    (flight2 / "proc").mkdir()

    # Add special folders to ignore
    (camp1 / "base").mkdir()
    (camp1 / "calibration").mkdir()

    # Add telescope_data campaign to ignore
    telescope = campaigns_dir / "telescope_data"
    telescope.mkdir()

    # Add a file (not directory) to test filtering
    (campaigns_dir / "readme.txt").write_text("test")

    return base


class TestPathLoaderInitialization:
    """Test PathLoader initialization."""

    def test_init_with_string_path(self, tmp_path):
        """Test initialization with string path."""
        loader = PathLoader(str(tmp_path))
        assert loader.base_data_path == Path(tmp_path)
        assert isinstance(loader.base_data_path, Path)

    def test_init_with_path_object(self, tmp_path):
        """Test initialization with Path object."""
        loader = PathLoader(tmp_path)
        assert loader.base_data_path == tmp_path
        assert isinstance(loader.base_data_path, Path)


class TestLoadAllFlights:
    """Test loading all flights."""

    def test_load_all_flights_returns_list(self, mock_campaign_structure):
        """Test load_all_flights returns list of flight dicts."""
        base, _ = mock_campaign_structure
        loader = PathLoader(base)

        flights = loader.load_all_flights()

        assert isinstance(flights, list)
        assert len(flights) == 2  # Two flights created in fixture

    def test_load_all_flights_structure(self, mock_campaign_structure):
        """Test that flight dicts have correct structure."""
        base, _ = mock_campaign_structure
        loader = PathLoader(base)

        flights = loader.load_all_flights()
        flight = flights[0]

        # Check all required keys exist
        assert "campaign_name" in flight
        assert "flight_name" in flight
        assert "flight_date" in flight
        assert "takeoff_datetime" in flight
        assert "landing_datetime" in flight
        assert "drone_data_folder_path" in flight
        assert "aux_data_folder_path" in flight
        assert "processed_data_folder_path" in flight

    def test_load_all_flights_paths_are_paths(self, mock_campaign_structure):
        """Test that returned folder paths are Path objects."""
        base, _ = mock_campaign_structure
        loader = PathLoader(base)

        flights = loader.load_all_flights()
        flight = flights[0]

        # Paths should be Path objects or strings that can be converted
        drone_path = Path(flight["drone_data_folder_path"])
        aux_path = Path(flight["aux_data_folder_path"])
        proc_path = Path(flight["processed_data_folder_path"])

        assert drone_path.exists()
        assert aux_path.exists()
        assert proc_path.exists()

    def test_load_all_flights_empty_campaigns_dir(self, tmp_path):
        """Test load_all_flights with no campaigns directory."""
        loader = PathLoader(tmp_path)
        flights = loader.load_all_flights()
        assert flights == []

    def test_load_all_flights_none_base_path(self):
        """Test load_all_flights with None base_data_path."""
        loader = PathLoader(None)
        flights = loader.load_all_flights()
        assert flights == []

    def test_load_all_flights_filters_telescope_data(self, mock_multi_campaign_structure):
        """Test that telescope_data campaign is filtered out."""
        base = mock_multi_campaign_structure
        loader = PathLoader(base)

        flights = loader.load_all_flights()

        # Should have flights from 202511 and 202512, but not telescope_data
        campaign_names = [f["campaign_name"] for f in flights]
        assert "telescope_data" not in campaign_names
        assert "202511" in campaign_names
        assert "202512" in campaign_names

    def test_load_all_flights_filters_base_calibration(self, mock_multi_campaign_structure):
        """Test that base and calibration folders are filtered out."""
        base = mock_multi_campaign_structure
        loader = PathLoader(base)

        flights = loader.load_all_flights()

        # Flight names should not include base or calibration
        flight_names = [f["flight_name"] for f in flights]
        assert "base" not in flight_names
        assert "calibration" not in flight_names

    def test_load_all_flights_parses_datetime(self, mock_campaign_structure):
        """Test that flight datetime is parsed correctly."""
        base, _ = mock_campaign_structure
        loader = PathLoader(base)

        flights = loader.load_all_flights()
        flight = flights[0]

        # Check that datetime was parsed
        takeoff = flight["takeoff_datetime"]
        assert isinstance(takeoff, str)  # ISO format string

        # Parse and validate
        dt = datetime.fromisoformat(takeoff)
        assert dt.year == 2025
        assert dt.month == 12
        assert dt.day == 8


class TestLoadSingleFlight:
    """Test single flight loading."""

    def test_load_single_flight_by_name(self, mock_campaign_structure):
        """Test load_single_flight with flight_name."""
        base, _ = mock_campaign_structure
        loader = PathLoader(base)

        flight = loader.load_single_flight(flight_name="flight_20251208_1506")

        assert flight is not None
        assert flight["flight_name"] == "flight_20251208_1506"

    def test_load_single_flight_nonexistent(self, mock_campaign_structure):
        """Test load_single_flight with nonexistent flight."""
        base, _ = mock_campaign_structure
        loader = PathLoader(base)

        flight = loader.load_single_flight(flight_name="nonexistent_flight")

        assert flight is None

    def test_load_single_flight_no_args_raises(self, mock_campaign_structure):
        """Test that load_single_flight raises when no args provided."""
        base, _ = mock_campaign_structure
        loader = PathLoader(base)

        with pytest.raises(ValueError, match="Either flight_id or flight_name"):
            loader.load_single_flight()

    def test_load_single_flight_returns_correct_paths(self, mock_campaign_structure):
        """Test that load_single_flight returns correct folder paths."""
        base, flight_path = mock_campaign_structure
        loader = PathLoader(base)

        flight = loader.load_single_flight(flight_name="flight_20251208_1506")

        assert flight is not None

        # Verify paths exist
        drone_path = Path(flight["drone_data_folder_path"])
        aux_path = Path(flight["aux_data_folder_path"])

        assert drone_path.exists()
        assert aux_path.exists()
        assert drone_path.name == "drone"
        assert aux_path.name == "aux"


class TestBuildFlightDictFromFilesystem:
    """Test _build_flight_dict_from_filesystem helper."""

    def test_build_flight_dict_valid_flight(self, mock_campaign_structure):
        """Test building flight dict from valid flight path."""
        base, flight_path = mock_campaign_structure
        loader = PathLoader(base)

        flight_dict = loader._build_flight_dict_from_filesystem(
            campaign_name="202511",
            date_folder="20251208",
            flight_name="flight_20251208_1506",
            flight_path=flight_path,
        )

        assert flight_dict is not None
        assert flight_dict["campaign_name"] == "202511"
        assert flight_dict["flight_name"] == "flight_20251208_1506"
        assert flight_dict["flight_date"] == "20251208"

    def test_build_flight_dict_invalid_datetime_format(self, tmp_path):
        """Test building flight dict with invalid datetime in name."""
        loader = PathLoader(tmp_path)

        # Invalid flight name format
        flight_dict = loader._build_flight_dict_from_filesystem(
            campaign_name="202511",
            date_folder="20251208",
            flight_name="invalid_name",
            flight_path=tmp_path,
        )

        # Should return None for invalid format
        assert flight_dict is None


class TestLoadAllCampaignFlights:
    """Test load_all_campaign_flights method."""

    def test_load_all_campaign_flights_raises_if_campaign_provided(self, mock_campaign_structure):
        """Test that providing campaign_name raises ValueError."""
        base, _ = mock_campaign_structure
        loader = PathLoader(base)

        # Current implementation raises if campaign_name is provided
        with pytest.raises(ValueError):
            loader.load_all_campaign_flights(campaign_name="202511")
