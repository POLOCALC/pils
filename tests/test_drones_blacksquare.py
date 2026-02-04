"""Test suite for BlackSquareDrone module following TDD methodology."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pils.drones.BlackSquareDrone import (
    BlackSquareDrone,
    get_leapseconds,
    messages_to_df,
    read_msgs,
)


class TestReadMsgs:
    """Test suite for reading ArduPilot log messages."""

    @pytest.fixture
    def sample_ardupilot_log(self, tmp_path):
        """Create sample ArduPilot log file."""
        log_content = """FMT, 128, 1, FMT, BBnNZ, Type,Length,Name,Format,Columns
FMT, 129, 2, GPS, QBIHBcLLeeee, TimeUS,Status,GMS,GWk,NSats,HDop,Lat,Lng,Alt,Spd
GPS, 1000000, 3, 123456, 2200, 8, 120, 407128000, -740060000, 10050, 250
GPS, 2000000, 3, 124456, 2200, 8, 120, 407129000, -740061000, 10100, 250
FMT, 130, 3, IMU, QffffffIIfBB, TimeUS,GyrX,GyrY,GyrZ,AccX,AccY,AccZ,EG,EA,T,GH,AH
IMU, 1000000, 0.01, -0.02, 0.03, 9.8, 0.1, -0.1, 1, 1, 25.5, 1, 1
IMU, 2000000, 0.02, -0.03, 0.04, 9.9, 0.2, -0.2, 1, 1, 25.6, 1, 1"""

        log_path = tmp_path / "test.log"
        log_path.write_text(log_content)
        return log_path

    def test_read_msgs_success(self, sample_ardupilot_log):
        """Test successful reading of ArduPilot log."""
        result = read_msgs(sample_ardupilot_log)
        assert isinstance(result, dict)
        assert "GPS" in result
        assert "IMU" in result

    def test_read_msgs_gps_data(self, sample_ardupilot_log):
        """Test GPS data parsing."""
        result = read_msgs(sample_ardupilot_log)
        gps_df = result["GPS"]
        assert isinstance(gps_df, pl.DataFrame)
        assert gps_df.shape[0] == 2  # Two GPS messages

    def test_read_msgs_multiple_message_types(self, sample_ardupilot_log):
        """Test parsing multiple message types."""
        result = read_msgs(sample_ardupilot_log)
        assert len(result) >= 2  # GPS and IMU at minimum

    def test_read_msgs_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_msgs("nonexistent.log")


class TestMessagesToDF:
    """Test suite for converting messages to DataFrame."""

    def test_messages_to_df_simple(self):
        """Test converting simple messages to DataFrame."""
        messages = [
            ["100", "200", "300"],
            ["101", "201", "301"],
        ]
        columns = ["col1", "col2", "col3"]
        format_str = "III"  # Three uint32

        df = messages_to_df(messages, columns, format_str)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (2, 3)

    def test_messages_to_df_with_floats(self):
        """Test converting messages with float values."""
        messages = [
            ["1.5", "2.5", "3.5"],
            ["4.5", "5.5", "6.5"],
        ]
        columns = ["x", "y", "z"]
        format_str = "fff"  # Three floats

        df = messages_to_df(messages, columns, format_str)
        assert df.shape == (2, 3)

    def test_messages_to_df_mixed_types(self):
        """Test converting messages with mixed types."""
        messages = [
            ["100", "1.5", "text"],
            ["200", "2.5", "more"],
        ]
        columns = ["int_col", "float_col", "str_col"]
        format_str = "IfN"  # uint32, float, char[16]

        df = messages_to_df(messages, columns, format_str)
        assert df.shape[0] == 2
        assert len(df.columns) == 3

    def test_messages_to_df_empty(self):
        """Test converting empty message list."""
        messages = []
        columns = ["col1"]
        format_str = "I"

        df = messages_to_df(messages, columns, format_str)
        assert df.shape[0] == 0


class TestGetLeapseconds:
    """Test suite for leap second calculation."""

    def test_get_leapseconds_2020(self):
        """Test leap seconds for year 2020."""
        result = get_leapseconds(2020, 1)
        assert isinstance(result, int)
        assert result >= 18  # At least 18 leap seconds by 2020

    def test_get_leapseconds_2024(self):
        """Test leap seconds for year 2024."""
        result = get_leapseconds(2024, 6)
        assert isinstance(result, int)
        assert result >= 18

    def test_get_leapseconds_different_months(self):
        """Test leap seconds for different months in same year."""
        jan = get_leapseconds(2020, 1)
        dec = get_leapseconds(2020, 12)
        # Should be the same year
        assert jan == dec


class TestBlackSquareDrone:
    """Test suite for BlackSquareDrone class."""

    @pytest.fixture
    def sample_log_file(self, tmp_path):
        """Create comprehensive sample log file."""
        log_content = """FMT, 128, 1, FMT, BBnNZ, Type,Length,Name,Format,Columns
FMT, 129, 2, GPS, QBIHBcLLeeee, TimeUS,Status,GMS,GWk,NSats,HDop,Lat,Lng,Alt,Spd
FMT, 130, 3, IMU, QffffffIIfBB, TimeUS,GyrX,GyrY,GyrZ,AccX,AccY,AccZ,EG,EA,T,GH,AH
FMT, 131, 4, BARO, QffcfI, TimeUS,Alt,Press,Temp,CRt,SMS
FMT, 132, 5, MAG, QhhhhhBI, TimeUS,MagX,MagY,MagZ,OfsX,OfsY,OfsZ,MOfsX
FMT, 133, 6, BAT, QfffffBBHB, TimeUS,Volt,Curr,CurrTot,EnrgTot,Temp,Res,RemPct,I,N
FMT, 134, 7, ATT, QccccCCC, TimeUS,Roll,Pitch,Yaw,Error,AEKF,Flags,Health
FMT, 135, 8, RCOU, QHHHHHHHHHHHHHH, TimeUS,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14
FMT, 136, 9, POS, QLLfff, TimeUS,Lat,Lng,Alt,RelH,RelHAlt
FMT, 137, 10, GPA, QBBB, TimeUS,VDop,HAcc,VAcc
FMT, 138, 11, PARM, QNf, TimeUS,Name,Value
GPS, 1000000, 3, 123456, 2200, 8, 120, 407128000, -740060000, 10050, 250
GPS, 2000000, 3, 124456, 2200, 8, 120, 407129000, -740061000, 10100, 250
IMU, 1000000, 0.01, -0.02, 0.03, 9.8, 0.1, -0.1, 1, 1, 25.5, 1, 1
BARO, 1000000, 100.5, 1013.25, 20.5, 0, 0
MAG, 1000000, 100, 200, -300, 0, 0, 0, 0
BAT, 1000000, 12.5, 5.2, 100.0, 150.0, 25.0, 0, 85, 0, 1
ATT, 1000000, 5, -2, 180, 0, 0, 0, 0
RCOU, 1000000, 1500, 1500, 1500, 1000, 1500, 1500, 1500, 1500, 0, 0, 0, 0, 0, 0
POS, 1000000, 407128000, -740060000, 100.5, 50.0, 50.0
GPA, 1000000, 150, 200, 300
PARM, 1000000, b'SYSID_THISMAV', 1.0"""

        log_path = tmp_path / "comprehensive.log"
        log_path.write_text(log_content)
        return log_path

    def test_init_with_string_path(self, sample_log_file):
        """Test initialization with string path."""
        drone = BlackSquareDrone(str(sample_log_file))
        assert drone.path == str(sample_log_file)
        assert drone.data is None

    def test_init_with_path_object(self, sample_log_file):
        """Test initialization with Path object."""
        drone = BlackSquareDrone(sample_log_file)
        assert drone.path == sample_log_file

    def test_load_data_success(self, sample_log_file):
        """Test successful data loading."""
        drone = BlackSquareDrone(sample_log_file)
        drone.load_data()
        assert drone.data is not None
        assert isinstance(drone.data, dict)
        assert drone.gps is not None
        assert drone.imu is not None

    def test_load_data_all_sensors(self, sample_log_file):
        """Test that all expected sensors are loaded."""
        drone = BlackSquareDrone(sample_log_file)
        drone.load_data()
        assert drone.barometer is not None
        assert drone.magnetometer is not None
        assert drone.batteries is not None
        assert drone.attitude is not None
        assert drone.pwm is not None

    def test_compute_datetime(self, sample_log_file):
        """Test datetime computation from GPS."""
        drone = BlackSquareDrone(sample_log_file)
        drone.load_data()
        drone.compute_datetime()
        assert drone.datetime is not None
        assert isinstance(drone.datetime, pl.Series)
        assert len(drone.datetime) == drone.gps.shape[0]

    def test_compute_datetime_values(self, sample_log_file):
        """Test that computed datetime values are reasonable."""
        drone = BlackSquareDrone(sample_log_file)
        drone.load_data()
        drone.compute_datetime()
        # Check that datetime is after GPS epoch
        gps_epoch = datetime(1980, 1, 6)
        first_dt = drone.datetime[0]
        assert first_dt > gps_epoch

    def test_params_cleaning(self, sample_log_file):
        """Test that PARM names are cleaned properly."""
        drone = BlackSquareDrone(sample_log_file)
        drone.load_data()
        assert drone.params is not None
        # Check that b' prefix is removed
        names = drone.params["Name"].to_list()
        assert all(not name.startswith("b'") for name in names)


class TestBlackSquareDroneLogging:
    """Test suite to verify logging instead of print statements."""

    def test_parse_error_logged(self, caplog, tmp_path):
        """Test that parse errors are logged instead of printed."""
        # Create log with invalid message format
        bad_log = tmp_path / "bad.log"
        log_content = """FMT, 128, 1, FMT, BBnNZ, Type,Length,Name,Format,Columns
FMT, 129, 2, GPS, QBI, TimeUS,Status,GMS
GPS, invalid, data, here, extra, columns"""
        bad_log.write_text(log_content)

        with caplog.at_level(logging.WARNING):
            result = read_msgs(bad_log)

        # Should log warning about failed parsing
        assert any("Failed to parse" in record.message for record in caplog.records)

    def test_no_print_statements(self, tmp_path, capsys):
        """Test that no print statements are used (only logging)."""
        # Create log with potential error
        log_file = tmp_path / "test.log"
        log_content = """FMT, 128, 1, FMT, BBnNZ, Type,Length,Name,Format,Columns
BADMSG, 1, 2, 3"""
        log_file.write_text(log_content)

        read_msgs(log_file)
        captured = capsys.readouterr()
        # Should not print warnings, only log them
        assert "Warning" not in captured.out or len(captured.out.strip()) == 0
