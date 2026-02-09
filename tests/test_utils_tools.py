"""
Tests for pils.utils.tools module.

Following TDD methodology - these tests are written first and will fail initially.
"""

import datetime
from pathlib import Path

import polars as pl
import pytest

from pils.utils import tools


class TestReadLogTime:
    """Test the read_log_time function."""

    def test_read_log_time_success(self, tmp_path):
        """Test successful timestamp extraction from log file."""
        log_file = tmp_path / "test.log"
        log_content = """2025/12/08 14:30:45.123456 [INFO] System startup
2025/12/08 14:30:46.234567 [INFO] Data acquisition started
2025/12/08 14:30:47.345678 [INFO] Sensor initialized
"""
        log_file.write_text(log_content)

        tstart, date = tools.read_log_time("Data acquisition", str(log_file))

        assert tstart is not None
        assert isinstance(tstart, datetime.datetime)
        assert tstart.year == 2025
        assert tstart.month == 12
        assert tstart.day == 8
        assert date == datetime.date(2025, 12, 8)

    def test_read_log_time_keyphrase_not_found(self, tmp_path):
        """Test when keyphrase is not in log file."""
        log_file = tmp_path / "test.log"
        log_content = "2025/12/08 14:30:45.123456 [INFO] System startup\n"
        log_file.write_text(log_content)

        tstart, date = tools.read_log_time("NonExistent", str(log_file))

        assert tstart is None
        assert date is None

    def test_read_log_time_with_path_object(self, tmp_path):
        """Test that function accepts Path object."""
        log_file = tmp_path / "test.log"
        log_content = "2025/12/08 14:30:45.123456 [INFO] Test message\n"
        log_file.write_text(log_content)

        # Pass Path object instead of string
        tstart, date = tools.read_log_time("Test message", log_file)

        assert tstart is not None
        assert isinstance(tstart, datetime.datetime)


class TestDropNanAndZeroCols:
    """Test the drop_nan_and_zero_cols function."""

    def test_drop_all_nan_column(self):
        """Test dropping column with all NaN values."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [None, None, None], "c": [4, 5, 6]})

        result = tools.drop_nan_and_zero_cols(df)

        assert "a" in result.columns
        assert "b" not in result.columns
        assert "c" in result.columns

    def test_drop_all_zero_column(self):
        """Test dropping column with all zero values."""
        df = pl.DataFrame({"x": [1, 2, 3], "y": [0, 0, 0], "z": [4, 5, 6]})

        result = tools.drop_nan_and_zero_cols(df)

        assert "x" in result.columns
        assert "y" not in result.columns
        assert "z" in result.columns

    def test_keep_mixed_column(self):
        """Test keeping column with mixed values."""
        df = pl.DataFrame({"a": [0, 1, 2], "b": [None, 1, 2], "c": [1, 2, 3]})

        result = tools.drop_nan_and_zero_cols(df)

        # All columns should be kept (they have at least one non-null/non-zero)
        assert "a" in result.columns
        assert "b" in result.columns
        assert "c" in result.columns

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pl.DataFrame()
        result = tools.drop_nan_and_zero_cols(df)
        assert result.shape == (0, 0)

    def test_non_numeric_columns(self):
        """Test with non-numeric columns (should not be dropped even if empty)."""
        df = pl.DataFrame({"text": ["a", "b", "c"], "numbers": [1, 2, 3]})

        result = tools.drop_nan_and_zero_cols(df)

        assert "text" in result.columns
        assert "numbers" in result.columns


class TestGetPathFromKeyword:
    """Test the get_path_from_keyword function."""

    def test_single_file_match(self, tmp_path):
        """Test finding a single file matching keyword."""
        # Create directory structure
        (tmp_path / "data").mkdir()
        test_file = tmp_path / "data" / "sensor_gps.csv"
        test_file.write_text("data")

        result = tools.get_path_from_keyword(str(tmp_path), "gps")

        assert result is not None
        assert isinstance(result, str)
        assert "sensor_gps.csv" in result

    def test_multiple_file_matches(self, tmp_path):
        """Test finding multiple files matching keyword."""
        (tmp_path / "data1").mkdir()
        (tmp_path / "data2").mkdir()
        (tmp_path / "data1" / "test_file.csv").write_text("data")
        (tmp_path / "data2" / "test_data.csv").write_text("data")

        result = tools.get_path_from_keyword(str(tmp_path), "test")

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 2

    def test_no_match(self, tmp_path):
        """Test when no files match keyword."""
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "sensor.csv").write_text("data")

        result = tools.get_path_from_keyword(str(tmp_path), "nonexistent")

        assert result is None

    def test_nested_directories(self, tmp_path):
        """Test searching in nested directory structure."""
        nested = tmp_path / "level1" / "level2" / "level3"
        nested.mkdir(parents=True)
        (nested / "deep_file.txt").write_text("data")

        result = tools.get_path_from_keyword(str(tmp_path), "deep")

        assert result is not None
        assert "deep_file.txt" in result

    def test_path_object_input(self, tmp_path):
        """Test function accepts Path object as input."""
        (tmp_path / "test.txt").write_text("data")

        result = tools.get_path_from_keyword(tmp_path, "test")

        assert result is not None


class TestIsAsciiFile:
    """Test the is_ascii_file function."""

    def test_ascii_file_bytes(self):
        """Test with valid ASCII bytes."""
        file_bytes = b"Hello World 123"
        assert tools.is_ascii_file(file_bytes) is True

    def test_non_ascii_file_bytes(self):
        """Test with non-ASCII bytes."""
        file_bytes = b"\xff\xfe\xfd"
        assert tools.is_ascii_file(file_bytes) is False

    def test_empty_bytes(self):
        """Test with empty bytes."""
        file_bytes = b""
        assert tools.is_ascii_file(file_bytes) is True


class TestGetLogpathFromDatapath:
    """Test the get_logpath_from_datapath function."""

    def test_success_case(self, tmp_path):
        """Test successful log path retrieval."""
        # Create structure: aux/sensors/gps.csv and aux/sensor_file.log
        aux_dir = tmp_path / "aux"
        sensors_dir = aux_dir / "sensors"
        sensors_dir.mkdir(parents=True)

        data_file = sensors_dir / "gps.csv"
        data_file.write_text("data")

        log_file = aux_dir / "sensor_file.log"
        log_file.write_text("log")

        result = tools.get_logpath_from_datapath(str(data_file))

        assert result is not None
        assert isinstance(result, Path)
        assert result.name == "sensor_file.log"
        assert result.exists()

    def test_datapath_not_exists(self, tmp_path):
        """Test with non-existent datapath."""
        fake_path = tmp_path / "nonexistent" / "file.csv"

        with pytest.raises(FileNotFoundError, match="does not exist"):
            tools.get_logpath_from_datapath(str(fake_path))

    def test_no_log_file(self, tmp_path):
        """Test when no log file exists."""
        aux_dir = tmp_path / "aux"
        sensors_dir = aux_dir / "sensors"
        sensors_dir.mkdir(parents=True)

        data_file = sensors_dir / "gps.csv"
        data_file.write_text("data")

        with pytest.raises(FileNotFoundError, match="No log file found"):
            tools.get_logpath_from_datapath(str(data_file))

    def test_multiple_log_files(self, tmp_path):
        """Test when multiple log files exist."""
        aux_dir = tmp_path / "aux"
        sensors_dir = aux_dir / "sensors"
        sensors_dir.mkdir(parents=True)

        data_file = sensors_dir / "gps.csv"
        data_file.write_text("data")

        (aux_dir / "sensor1_file.log").write_text("log1")
        (aux_dir / "sensor2_file.log").write_text("log2")

        with pytest.raises(FileExistsError, match="Multiple log files found"):
            tools.get_logpath_from_datapath(str(data_file))

    def test_path_object_input(self, tmp_path):
        """Test function accepts Path object as input."""
        aux_dir = tmp_path / "aux"
        sensors_dir = aux_dir / "sensors"
        sensors_dir.mkdir(parents=True)

        data_file = sensors_dir / "gps.csv"
        data_file.write_text("data")

        log_file = aux_dir / "sensor_file.log"
        log_file.write_text("log")

        # Pass Path object
        result = tools.get_logpath_from_datapath(data_file)

        assert result is not None
        assert result.name == "sensor_file.log"


class TestFahrenheitToCelsius:
    """Test the fahrenheit_to_celsius function."""

    def test_freezing_point(self):
        """Test conversion of water freezing point."""
        result = tools.fahrenheit_to_celsius(32.0)
        assert abs(result - 0.0) < 0.01

    def test_boiling_point(self):
        """Test conversion of water boiling point."""
        result = tools.fahrenheit_to_celsius(212.0)
        assert abs(result - 100.0) < 0.01

    def test_negative_temperature(self):
        """Test conversion of negative temperature."""
        result = tools.fahrenheit_to_celsius(-40.0)
        assert abs(result - (-40.0)) < 0.01

    def test_room_temperature(self):
        """Test conversion of room temperature."""
        result = tools.fahrenheit_to_celsius(68.0)
        assert abs(result - 20.0) < 0.01
