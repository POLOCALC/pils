"""Tests for StoutLoader."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from pils.loader.stout import StoutLoader


@pytest.fixture
def mock_stout_available():
    """Mock STOUT availability."""
    with patch("pils.loader.stout.importlib") as mock_importlib:
        # Simulate successful import
        mock_config = MagicMock()
        mock_config.MAIN_DATA_PATH = "/data"
        mock_campaign_service = MagicMock()

        def mock_import(name):
            if name == "stout.config":
                return type("Module", (), {"Config": mock_config})()
            elif name == "stout.services.campaigns":
                return type("Module", (), {"CampaignService": lambda: mock_campaign_service})()
            raise ImportError(f"Module {name} not found")

        mock_importlib.import_module.side_effect = mock_import
        yield mock_campaign_service


@pytest.fixture
def mock_campaign_structure(tmp_path):
    """Create mock campaign directory structure for filesystem fallback."""
    base = tmp_path / "data"
    campaigns = base / "campaigns"

    # Create campaign structure: campaigns/202511/20251208/flight_20251208_1506
    campaign_dir = campaigns / "202511" / "20251208"
    flight_dir = campaign_dir / "flight_20251208_1506"

    flight_dir.mkdir(parents=True, exist_ok=True)
    (flight_dir / "drone").mkdir()
    (flight_dir / "aux").mkdir()
    (flight_dir / "proc").mkdir()

    # Create GPS sensor data
    gps_dir = flight_dir / "drone" / "gps"
    gps_dir.mkdir(parents=True)
    (gps_dir / "gps_data.csv").write_text("timestamp,lat,lon\n1000,40.7128,-74.0060\n")

    # Create another flight for multi-flight tests
    flight_dir2 = campaign_dir / "flight_20251208_1530"
    flight_dir2.mkdir(parents=True)
    (flight_dir2 / "drone").mkdir()
    (flight_dir2 / "aux").mkdir()
    (flight_dir2 / "proc").mkdir()

    return base


class TestStoutLoaderInitialization:
    """Test StoutLoader initialization."""

    def test_init_attempts_stout_import(self):
        """Test initialization attempts to import STOUT."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            # Simulate STOUT not available
            mock_import.side_effect = ImportError("No module named 'stout'")

            loader = StoutLoader()

            # Should fall back to filesystem mode
            assert loader.campaign_service is None


class TestLoadAllFlightsFromFilesystem:
    """Test _load_all_flights_from_filesystem method."""

    def test_load_all_flights_returns_list(self, mock_campaign_structure):
        """Test loading all flights from filesystem returns list."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            flights = loader._load_all_flights_from_filesystem()

            assert isinstance(flights, list)
            assert len(flights) == 2  # Two flights created in fixture

    def test_load_all_flights_contains_required_fields(self, mock_campaign_structure):
        """Test loaded flights contain required fields."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            flights = loader._load_all_flights_from_filesystem()

            assert len(flights) > 0
            flight = flights[0]
            assert "campaign_name" in flight
            assert "flight_name" in flight
            assert "flight_date" in flight
            assert "drone_data_folder_path" in flight
            assert "aux_data_folder_path" in flight
            assert "processed_data_folder_path" in flight

    def test_load_all_flights_returns_empty_when_no_campaigns_dir(self, tmp_path):
        """Test returns empty list when campaigns directory missing."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = tmp_path  # No campaigns directory

            flights = loader._load_all_flights_from_filesystem()

            assert flights == []

    def test_load_all_flights_skips_non_directory_entries(self, mock_campaign_structure):
        """Test that non-directory entries are skipped."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            # Create a file in campaigns directory (should be skipped)
            campaigns_dir = mock_campaign_structure / "campaigns"
            (campaigns_dir / "readme.txt").write_text("This is a file")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            flights = loader._load_all_flights_from_filesystem()

            # Should still find the 2 flights, ignoring the file
            assert len(flights) == 2


class TestLoadSingleFlightFromFilesystem:
    """Test _load_single_flight_from_filesystem method."""

    def test_load_single_flight_by_name(self, mock_campaign_structure):
        """Test loading single flight by flight_name."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            flight = loader._load_single_flight_from_filesystem(flight_name="flight_20251208_1506")

            assert flight is not None
            assert flight["flight_name"] == "flight_20251208_1506"

    def test_load_single_flight_returns_none_when_not_found(self, mock_campaign_structure):
        """Test returns None when flight not found."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            flight = loader._load_single_flight_from_filesystem(flight_name="nonexistent_flight")

            assert flight is None


class TestBuildFlightDictFromFilesystem:
    """Test _build_flight_dict_from_filesystem method."""

    def test_build_flight_dict_creates_correct_structure(self):
        """Test building flight dict creates correct structure."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()

            flight_dict = loader._build_flight_dict_from_filesystem(
                campaign_name="202511",
                date_folder="20251208",
                flight_name="flight_20251208_1506",
                flight_path="/data/campaigns/202511/20251208/flight_20251208_1506",
            )

            assert flight_dict is not None
            assert flight_dict["campaign_name"] == "202511"
            assert flight_dict["flight_name"] == "flight_20251208_1506"
            assert flight_dict["flight_date"] == "20251208"
            assert "drone_data_folder_path" in flight_dict
            assert "aux_data_folder_path" in flight_dict
            assert "processed_data_folder_path" in flight_dict

    def test_build_flight_dict_handles_invalid_date_format(self):
        """Test handles invalid date format gracefully."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()

            flight_dict = loader._build_flight_dict_from_filesystem(
                campaign_name="202511",
                date_folder="invalid_date",
                flight_name="flight_test",
                flight_path="/data/test",
            )

            # Should return None for invalid date
            assert flight_dict is None


class TestListFilesRecursive:
    """Test _list_files_recursive method."""

    def test_list_files_recursive_finds_all_files(self, mock_campaign_structure):
        """Test recursive file listing finds all files."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()

            gps_dir = (
                mock_campaign_structure
                / "campaigns"
                / "202511"
                / "20251208"
                / "flight_20251208_1506"
                / "drone"
                / "gps"
            )

            files = loader._list_files_recursive(str(gps_dir))

            assert len(files) > 0
            assert any("gps_data.csv" in f for f in files)

    def test_list_files_recursive_returns_empty_for_nonexistent_dir(self):
        """Test returns empty list for nonexistent directory."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()

            files = loader._list_files_recursive("/nonexistent/directory")

            assert files == []


class TestGetCampaignListFromFilesystem:
    """Test _get_campaigns_from_filesystem method."""

    def test_get_campaign_list_returns_campaigns(self, mock_campaign_structure):
        """Test getting campaign list from filesystem."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            campaigns = loader._get_campaigns_from_filesystem()

            assert len(campaigns) == 1
            assert campaigns[0]["name"] == "202511"

    def test_get_campaign_list_returns_empty_when_no_base_path(self):
        """Test returns empty list when base_data_path is None."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = None

            campaigns = loader._get_campaigns_from_filesystem()

            assert campaigns == []


class TestPathlibUsage:
    """Test that pathlib is used instead of os.path."""

    def test_drone_data_folder_path_uses_pathlib(self, mock_campaign_structure):
        """Test that drone_data_folder_path is constructed with pathlib."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            flights = loader._load_all_flights_from_filesystem()

            assert len(flights) > 0
            flight = flights[0]

            # Path should contain proper separators (pathlib handles this)
            drone_path = flight["drone_data_folder_path"]
            assert "drone" in str(drone_path)

    def test_campaigns_dir_constructed_with_pathlib(self, mock_campaign_structure):
        """Test campaigns directory constructed with pathlib."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            # This should use pathlib internally
            campaigns = loader._get_campaigns_from_filesystem()

            assert len(campaigns) > 0


class TestLoadFlightsByDateFromFilesystem:
    """Test _load_flights_by_date_from_filesystem method."""

    def test_load_flights_by_date_filters_correctly(self, mock_campaign_structure):
        """Test loading flights by date range filters correctly."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            start_dt = datetime(2025, 12, 1, tzinfo=UTC)
            end_dt = datetime(2025, 12, 31, tzinfo=UTC)

            flights = loader._load_flights_by_date_from_filesystem(start_dt, end_dt)

            assert len(flights) == 2  # Both flights are in December 2025

    def test_load_flights_by_date_excludes_outside_range(self, mock_campaign_structure):
        """Test flights outside date range are excluded."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()
            loader.base_data_path = mock_campaign_structure

            start_dt = datetime(2026, 1, 1, tzinfo=UTC)
            end_dt = datetime(2026, 1, 31, tzinfo=UTC)

            flights = loader._load_flights_by_date_from_filesystem(start_dt, end_dt)

            assert len(flights) == 0  # No flights in January 2026


class TestCollectSpecificData:
    """Test _collect_specific_data method."""

    def test_collect_specific_data_returns_dict(self, mock_campaign_structure):
        """Test collecting specific data returns dictionary."""
        with patch("pils.loader.stout.importlib.import_module") as mock_import:
            mock_import.side_effect = ImportError("No module")

            loader = StoutLoader()

            flight = {
                "drone_data_folder_path": str(
                    mock_campaign_structure
                    / "campaigns"
                    / "202511"
                    / "20251208"
                    / "flight_20251208_1506"
                    / "drone"
                ),
                "aux_data_folder_path": str(
                    mock_campaign_structure
                    / "campaigns"
                    / "202511"
                    / "20251208"
                    / "flight_20251208_1506"
                    / "aux"
                ),
            }

            data = loader._collect_specific_data(flight, data_types=["gps"])

            assert isinstance(data, dict)
