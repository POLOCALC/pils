"""Tests for GPS sensor module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import polars as pl
import pytest

from pils.sensors.gps import GPS


class TestGPS:
    """Test suite for GPS class."""

    @pytest.fixture
    def gps_dir(self, tmp_path):
        """Create GPS directory with gps.bin file."""
        gps_path = tmp_path / "gps_data"
        gps_path.mkdir()
        gps_file = gps_path / "gps.bin"
        # Create empty binary file (actual content will be mocked)
        gps_file.write_bytes(b"")
        return gps_path

    @pytest.fixture
    def log_file(self, tmp_path):
        """Create log file with GPS start timestamp."""
        log_path = tmp_path / "flight.log"
        log_path.write_text(
            "2024-01-15 10:30:45 INFO:Sensor ZED-F9P started\n"
            "2024-01-15 10:30:50 INFO:GPS recording\n"
        )
        return log_path

    def test_init_with_path_finds_gps_bin(self, gps_dir):
        """Test GPS initialization finds gps.bin file."""
        with patch("pils.sensors.gps.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            gps = GPS(gps_dir)
            assert gps.data_path.name == "gps.bin"

    def test_init_sets_logpath(self, gps_dir, log_file):
        """Test GPS initialization with explicit logpath."""
        gps = GPS(gps_dir, logpath=log_file)
        assert gps.logpath == log_file

    def test_init_with_path_type(self, gps_dir, log_file):
        """Test GPS accepts Path type."""
        gps = GPS(gps_dir, logpath=log_file)
        assert isinstance(gps.data_path, Path)
        assert gps.tstart is None
        assert gps.data is None

    def test_init_finds_gps_case_insensitive(self, tmp_path):
        """Test GPS finds GPS.BIN (case insensitive)."""
        gps_path = tmp_path / "gps_data"
        gps_path.mkdir()
        gps_file = gps_path / "GPS.BIN"
        gps_file.write_bytes(b"")

        with patch("pils.sensors.gps.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            gps = GPS(gps_path)
            assert gps.data_path.name == "GPS.BIN"

    @patch("pils.sensors.gps.UBXReader")
    @patch("pils.sensors.gps.read_log_time")
    def test_load_data_creates_dataframe(
        self, mock_read_log, mock_ubx, gps_dir, log_file
    ):
        """Test load_data creates polars DataFrame."""
        # Mock log reading
        mock_read_log.return_value = (
            datetime(2024, 1, 15, 10, 30, 45),
            datetime(2024, 1, 15).date(),
        )

        # Mock UBX reader to return empty data
        mock_ubr_instance = Mock()
        mock_ubr_instance.__iter__ = Mock(return_value=iter([]))
        mock_ubx.return_value = mock_ubr_instance

        gps = GPS(gps_dir, logpath=log_file)
        gps.load_data()

        assert gps.data is not None
        assert isinstance(gps.data, pl.DataFrame)

    @patch("pils.sensors.gps.UBXReader")
    @patch("pils.sensors.gps.read_log_time")
    def test_load_data_with_freq_interpolation(
        self, mock_read_log, mock_ubx, gps_dir, log_file
    ):
        """Test load_data with frequency interpolation parameter."""
        # Mock log reading
        mock_read_log.return_value = (
            datetime(2024, 1, 15, 10, 30, 45),
            datetime(2024, 1, 15).date(),
        )

        # Mock UBX reader
        mock_ubr_instance = Mock()
        mock_ubr_instance.__iter__ = Mock(return_value=iter([]))
        mock_ubx.return_value = mock_ubr_instance

        gps = GPS(gps_dir, logpath=log_file)
        gps.load_data(freq_interpolation=10.0)

        assert gps.data is not None
        mock_ubx.assert_called_once()

    @patch("pils.sensors.gps.UBXReader")
    @patch("pils.sensors.gps.read_log_time")
    def test_load_data_returns_none_type(
        self, mock_read_log, mock_ubx, gps_dir, log_file
    ):
        """Test load_data return type is None."""
        mock_read_log.return_value = (
            datetime(2024, 1, 15, 10, 30, 45),
            datetime(2024, 1, 15).date(),
        )

        mock_ubr_instance = Mock()
        mock_ubr_instance.__iter__ = Mock(return_value=iter([]))
        mock_ubx.return_value = mock_ubr_instance

        gps = GPS(gps_dir, logpath=log_file)
        result = gps.load_data()

        assert result is None

    def test_path_attribute_is_path_type(self, gps_dir):
        """Test that data_path is a Path object."""
        with patch("pils.sensors.gps.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            gps = GPS(gps_dir)
            assert isinstance(gps.data_path, Path)

    @patch("pils.sensors.gps.UBXReader")
    @patch("pils.sensors.gps.read_log_time")
    def test_merge_nav_dataframes_with_empty_dict(
        self, mock_read_log, mock_ubx, gps_dir, log_file
    ):
        """Test _merge_nav_dataframes handles empty input."""
        gps = GPS(gps_dir, logpath=log_file)
        result = gps._merge_nav_dataframes({})
        assert result is None

    @patch("pils.sensors.gps.UBXReader")
    @patch("pils.sensors.gps.read_log_time")
    def test_merge_nav_dataframes_with_freq(
        self, mock_read_log, mock_ubx, gps_dir, log_file
    ):
        """Test _merge_nav_dataframes with frequency parameter."""
        gps = GPS(gps_dir, logpath=log_file)

        # Create simple nav dataframe for testing
        nav_df = pl.DataFrame(
            {"unix_time_ms": [1000, 2000, 3000], "lat": [40.0, 40.1, 40.2]}
        )

        result = gps._merge_nav_dataframes({"NAV-POSLLH": nav_df}, freq=10.0)

        assert result is not None
        assert isinstance(result, pl.DataFrame)
        assert "unix_time_ms" in result.columns
