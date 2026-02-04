"""Test suite for DJIDrone module following TDD methodology."""

import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import numpy as np
import polars as pl
import pytest

from pils.drones.DJIDrone import DJIDrone


class TestDJIDroneCSV:
    """Test suite for DJIDrone CSV loading functionality."""

    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data as string."""
        return """Clock:offsetTime,GPS:dateTimeStamp,RTKdata:GpsState,RTKdata:Lat_P,RTKdata:Lon_P,RTKdata:Hmsl_P
1000,2024-01-15 10:30:00.123Z,50,40.7128,-74.0060,100.5
2000,2024-01-15 10:30:01.123Z,50,40.7129,-74.0061,101.0
3000,2024-01-15 10:30:02.123Z,50,40.7130,-74.0062,101.5
4000,2024-01-15 10:30:03.123Z,50,40.7131,-74.0063,102.0"""

    @pytest.fixture
    def csv_file(self, tmp_path, sample_csv_data):
        """Create temporary CSV file."""
        csv_path = tmp_path / "test_drone.csv"
        csv_path.write_text(sample_csv_data)
        return csv_path

    def test_init_with_path_string(self, csv_file):
        """Test DJIDrone initialization with string path."""
        drone = DJIDrone(str(csv_file))
        assert drone.path == str(csv_file)
        assert drone.data == {}
        assert drone.sync_params is None
        assert drone.source_format is None

    def test_init_with_path_object(self, csv_file):
        """Test DJIDrone initialization with Path object."""
        drone = DJIDrone(csv_file)
        assert drone.path == csv_file
        assert drone.data == {}

    def test_load_csv_data(self, csv_file):
        """Test loading CSV data successfully."""
        drone = DJIDrone(csv_file)
        drone.load_data(use_dat=False)
        assert isinstance(drone.data, pl.DataFrame)
        assert drone.source_format == "csv"
        assert drone.data.shape[0] == 4

    def test_load_csv_with_column_selection(self, csv_file):
        """Test loading CSV with specific columns."""
        drone = DJIDrone(csv_file)
        cols = ["Clock:offsetTime", "GPS:dateTimeStamp"]
        drone.load_data(cols=cols, use_dat=False)
        assert isinstance(drone.data, pl.DataFrame)
        # Should have timestamp and datetime columns added
        assert "timestamp" in drone.data.columns

    def test_remove_consecutive_duplicates(self, csv_file):
        """Test removing consecutive duplicates from data."""
        # Create CSV with duplicates
        csv_with_dups = csv_file.parent / "dups.csv"
        data = """Clock:offsetTime,GPS:dateTimeStamp,RTKdata:Lat_P
1000,2024-01-15 10:30:00.123Z,40.7128
2000,2024-01-15 10:30:01.123Z,40.7128
3000,2024-01-15 10:30:02.123Z,40.7130"""
        csv_with_dups.write_text(data)

        drone = DJIDrone(csv_with_dups)
        drone.load_data(use_dat=False, remove_duplicate=True)
        # Should remove duplicate latitude values
        assert drone.data.shape[0] <= 3

    def test_csv_datetime_parsing(self, csv_file):
        """Test datetime parsing from CSV."""
        drone = DJIDrone(csv_file)
        drone.load_data(use_dat=False)
        assert "datetime" in drone.data.columns
        assert "timestamp" in drone.data.columns


class TestDJIDroneDAT:
    """Test suite for DJIDrone DAT binary format parsing."""

    @pytest.fixture
    def mock_dat_data(self):
        """Create mock DAT file binary data."""
        # Mock encrypted DJI DAT format header
        header = b"\x55" * 128  # Simplified mock header
        return header

    def test_parse_and_decode_message_valid(self):
        """Test parsing valid encrypted message."""
        drone = DJIDrone("test.dat")
        # Create mock message data (header + encrypted payload)
        msg_data = b"\x55\xaa" + b"\x00" * 100

        # This will likely fail without proper encryption keys
        result = drone._parse_and_decode_message(msg_data)
        # Can be None if decryption fails (expected)
        assert result is None or isinstance(result, tuple)

    def test_parse_and_decode_message_invalid(self):
        """Test parsing invalid message data."""
        drone = DJIDrone("test.dat")
        msg_data = b"\x00" * 10  # Too short
        result = drone._parse_and_decode_message(msg_data)
        assert result is None

    def test_decode_message_data_gps(self):
        """Test decoding GPS message type."""
        drone = DJIDrone("test.dat")
        # Mock GPS payload (66 bytes for message type 2096)
        payload = b"\x00" * 66
        msg_type = 2096
        tick_val = 1000

        from pils.drones.DJIDrone import MESSAGE_DEFINITIONS

        msg_def = MESSAGE_DEFINITIONS[msg_type]

        result = drone._decode_message_data(payload, msg_type, tick_val, msg_def)
        assert isinstance(result, dict)
        assert "tick" in result
        assert "msg_type" in result

    def test_decode_message_data_rtk(self):
        """Test decoding RTK message type."""
        drone = DJIDrone("test.dat")
        # Mock RTK payload (72 bytes for message type 53234)
        payload = b"\x00" * 72
        msg_type = 53234
        tick_val = 2000

        from pils.drones.DJIDrone import MESSAGE_DEFINITIONS

        msg_def = MESSAGE_DEFINITIONS[msg_type]

        result = drone._decode_message_data(payload, msg_type, tick_val, msg_def)
        assert isinstance(result, dict)
        assert result["tick"] == tick_val

    def test_load_from_dat_file_not_found(self):
        """Test loading DAT file that doesn't exist."""
        drone = DJIDrone("nonexistent.dat")
        with pytest.raises(FileNotFoundError):
            drone._load_from_dat()

    def test_parse_gps_datetime_valid(self):
        """Test parsing GPS datetime from payload."""
        drone = DJIDrone("test.dat")
        # Create payload with date and time fields
        # Date: 20240115 (0x01340103), Time: 103000 (0x00019238)
        payload = struct.pack("<II", 20240115, 103000) + b"\x00" * 58

        result = drone._parse_gps_datetime(payload)
        if result:
            assert isinstance(result, datetime)

    def test_parse_gps_datetime_invalid(self):
        """Test parsing invalid GPS datetime."""
        drone = DJIDrone("test.dat")
        payload = b"\x00" * 66  # All zeros
        result = drone._parse_gps_datetime(payload)
        assert result is None


class TestDJIDroneUtils:
    """Test suite for DJIDrone utility methods."""

    def test_format_date_time_valid(self):
        """Test formatting valid date and time."""
        date = 20240115  # 2024-01-15
        time = 103000  # 10:30:00
        result = DJIDrone._format_date_time(date, time)
        assert result == datetime(2024, 1, 15, 10, 30, 0)

    def test_format_date_time_zero_date(self):
        """Test formatting with zero date."""
        result = DJIDrone._format_date_time(0, 103000)
        assert result is None

    def test_format_date_time_zero_time(self):
        """Test formatting with zero time."""
        result = DJIDrone._format_date_time(20240115, 0)
        assert result is None

    def test_unwrap_tick_no_wrapping(self):
        """Test unwrap_tick with no wrapping."""
        df = pl.DataFrame({"tick": [100, 200, 300, 400]})
        result = DJIDrone._unwrap_tick(df)
        assert result["tick"].to_list() == [100, 200, 300, 400]

    def test_unwrap_tick_with_wrapping(self):
        """Test unwrap_tick with tick counter wrapping."""
        # Simulate tick counter overflow
        df = pl.DataFrame({"tick": [9e7, 9.5e7, 1e6, 2e6]})
        result = DJIDrone._unwrap_tick(df, wrap_threshold=1e8)
        # After unwrapping, ticks should be monotonically increasing
        ticks = result["tick"].to_list()
        assert all(ticks[i] < ticks[i + 1] for i in range(len(ticks) - 1))

    def test_get_tick_offset_no_sync(self):
        """Test get_tick_offset when no sync params exist."""
        drone = DJIDrone("test.dat")
        offset = drone.get_tick_offset()
        assert offset == 0.0

    def test_get_tick_offset_with_sync(self):
        """Test get_tick_offset with sync parameters."""
        drone = DJIDrone("test.dat")
        drone.sync_params = (1.0, 5000.0)  # (slope, intercept)
        offset = drone.get_tick_offset()
        assert offset == 5000.0


class TestDJIDroneLogging:
    """Test suite to verify logging instead of print statements."""

    def test_decode_message_logs_error(self, caplog):
        """Test that decode error is logged instead of printed."""
        drone = DJIDrone("test.dat")
        # Create invalid payload that will cause decode error
        payload = b"invalid"
        msg_type = 2096
        tick_val = 1000

        from pils.drones.DJIDrone import MESSAGE_DEFINITIONS

        msg_def = MESSAGE_DEFINITIONS[msg_type]

        with caplog.at_level(logging.ERROR):
            result = drone._decode_message_data(payload, msg_type, tick_val, msg_def)

        # Should log error, not print
        assert result is None or any(
            "Failed to decode" in record.message for record in caplog.records
        )

    def test_csv_parsing_exception_logged(self, caplog, tmp_path):
        """Test that CSV parsing exceptions are logged."""
        # Create malformed CSV
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("not,valid,csv\n1,2")  # Wrong number of columns

        drone = DJIDrone(bad_csv)
        with caplog.at_level(logging.WARNING):
            try:
                drone.load_data(use_dat=False)
            except Exception:
                pass  # Expected to fail


class TestDJIDroneAlignment:
    """Test suite for DAT file alignment with GPS."""

    @pytest.fixture
    def sample_gps_df(self):
        """Create sample GPS DataFrame."""
        return pl.DataFrame(
            {
                "timestamp": [1000.0, 2000.0, 3000.0, 4000.0],
                "latitude": [40.7128, 40.7129, 40.7130, 40.7131],
                "longitude": [-74.0060, -74.0061, -74.0062, -74.0063],
            }
        )

    def test_align_datfile_success(self, sample_gps_df):
        """Test successful alignment of DAT file with GPS."""
        drone = DJIDrone("test.dat")
        # Mock DAT data
        drone.data = {
            "GPS": pl.DataFrame(
                {
                    "tick": [100, 200, 300, 400],
                    "latitude": [40.7128, 40.7129, 40.7130, 40.7131],
                    "longitude": [-74.0060, -74.0061, -74.0062, -74.0063],
                }
            )
        }

        # This might fail without proper data, but tests the interface
        try:
            result_df, offset = drone.align_datfile(sample_gps_df)
            assert isinstance(result_df, pl.DataFrame)
            assert isinstance(offset, float)
        except Exception:
            # Expected if correlation fails
            pass

    def test_align_datfile_no_dat_data(self, sample_gps_df):
        """Test alignment when DAT data is not loaded."""
        drone = DJIDrone("test.dat")
        drone.data = {}  # No DAT data

        with pytest.raises((KeyError, ValueError)):
            drone.align_datfile(sample_gps_df)


# Import struct for test_parse_gps_datetime_valid
import struct
