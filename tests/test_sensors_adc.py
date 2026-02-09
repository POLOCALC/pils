"""Tests for ADC sensor module (pils/sensors/adc.py)."""

import struct
from unittest.mock import patch

import polars as pl
import pytest

from pils.sensors.adc import ADC, decode_adc_file_ascii, decode_adc_file_struct


class TestDecodeAdcFileStruct:
    """Test suite for decode_adc_file_struct function."""

    def test_decode_binary_adc_file(self, tmp_path):
        """Test decoding binary structured ADC file."""
        # Create binary ADC data (pattern: dqf = uint64, uint64, double)
        data = b""
        timestamps = [1000, 2000, 3000]
        reading_times = [100, 200, 300]
        amplitudes = [1.5, 2.5, 3.5]

        for ts, rt, amp in zip(timestamps, reading_times, amplitudes, strict=False):
            data += struct.pack("<dqf", ts, rt, amp)

        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(data)

        # Decode the file
        df = decode_adc_file_struct(adc_file)

        # Verify structure
        assert isinstance(df, pl.DataFrame)
        assert "timestamp" in df.columns
        assert "reading_time" in df.columns
        assert "amplitude" in df.columns
        assert "datetime" in df.columns
        assert df.shape[0] == 3

    def test_decode_binary_with_path_object(self, tmp_path):
        """Test that function accepts Path object."""
        data = struct.pack("<dqf", 1000.0, 100, 1.5)
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(data)

        df = decode_adc_file_struct(adc_file)
        assert isinstance(df, pl.DataFrame)

    def test_decode_binary_with_string_path(self, tmp_path):
        """Test that function accepts string path."""
        data = struct.pack("<dqf", 1000.0, 100, 1.5)
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(data)

        df = decode_adc_file_struct(str(adc_file))
        assert isinstance(df, pl.DataFrame)

    def test_datetime_conversion(self, tmp_path):
        """Test that datetime column is properly converted from timestamp."""
        timestamp = 1609459200  # 2021-01-01 00:00:00 UTC
        data = struct.pack("<dqf", timestamp, 100, 1.5)
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(data)

        df = decode_adc_file_struct(adc_file)
        assert "datetime" in df.columns
        assert df["datetime"].dtype == pl.Datetime


class TestDecodeAdcFileAscii:
    """Test suite for decode_adc_file_ascii function."""

    def test_decode_ascii_adc_file(self, tmp_path):
        """Test decoding ASCII ADC file."""
        # Create ASCII ADC data (timestamp in microseconds, amplitude as raw value)
        content = b"1000000 1024\n2000000 2048\n3000000 512\n"
        adc_file = tmp_path / "test_adc.txt"
        adc_file.write_bytes(content)

        # Decode with default gain=16
        df = decode_adc_file_ascii(adc_file, gain_config=16)

        # Verify structure
        assert isinstance(df, pl.DataFrame)
        assert "timestamp" in df.columns
        assert "amplitude" in df.columns
        assert df.shape[0] == 3

        # Verify timestamp conversion (microseconds to seconds)
        assert df["timestamp"][0] == pytest.approx(1.0, abs=0.01)
        assert df["timestamp"][1] == pytest.approx(2.0, abs=0.01)

    def test_gain_conversion(self, tmp_path):
        """Test that gain configuration affects amplitude conversion."""
        content = b"1000000 2048\n"  # Full scale value
        adc_file = tmp_path / "test_adc.txt"
        adc_file.write_bytes(content)

        # Test with gain=16 (0.256V range)
        df1 = decode_adc_file_ascii(adc_file, gain_config=16)
        amp1 = df1["amplitude"][0]

        # Test with gain=1 (4.096V range)
        df2 = decode_adc_file_ascii(adc_file, gain_config=1)
        amp2 = df2["amplitude"][0]

        # Gain=1 should give 16x larger amplitude than gain=16
        assert amp2 == pytest.approx(amp1 * 16, abs=0.1)

    def test_skip_empty_lines(self, tmp_path):
        """Test that empty lines are skipped."""
        content = b"1000000 1024\n\n2000000 2048\n"
        adc_file = tmp_path / "test_adc.txt"
        adc_file.write_bytes(content)

        df = decode_adc_file_ascii(adc_file)
        assert df.shape[0] == 2

    def test_skip_incomplete_lines(self, tmp_path):
        """Test that incomplete lines are skipped."""
        content = b"1000000 1024\n1234\n2000000 2048\n"
        adc_file = tmp_path / "test_adc.txt"
        adc_file.write_bytes(content)

        df = decode_adc_file_ascii(adc_file)
        assert df.shape[0] == 2

    def test_logging_on_parse_error(self, tmp_path, caplog):
        """Test that parse errors are logged instead of printed."""
        content = b"1000000 1024\nabc def\n2000000 2048\n"
        adc_file = tmp_path / "test_adc.txt"
        adc_file.write_bytes(content)

        with caplog.at_level("WARNING"):
            df = decode_adc_file_ascii(adc_file)

        # Should have logged warning about line 2
        assert df.shape[0] == 2
        assert any("line 2" in record.message.lower() for record in caplog.records)

    def test_logging_on_general_error(self, tmp_path, caplog):
        """Test that general decoding errors are logged."""
        # Create file with special characters that might cause decode issues
        content = b"1000000 1024\n\xff\xfe invalid\n2000000 2048\n"
        adc_file = tmp_path / "test_adc.txt"
        adc_file.write_bytes(content)

        with caplog.at_level("ERROR"):
            df = decode_adc_file_ascii(adc_file)

        # Should still decode the valid lines
        assert df.shape[0] == 2

    def test_path_string_and_path_object(self, tmp_path):
        """Test that function accepts both str and Path."""
        content = b"1000000 1024\n"
        adc_file = tmp_path / "test_adc.txt"
        adc_file.write_bytes(content)

        df1 = decode_adc_file_ascii(str(adc_file))
        df2 = decode_adc_file_ascii(adc_file)

        assert df1.shape == df2.shape


class TestADCClass:
    """Test suite for ADC class."""

    def test_adc_initialization(self, tmp_path):
        """Test ADC class initialization."""
        # Create dummy ADC file
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(b"dummy data")

        adc = ADC(tmp_path, logpath=None, gain_config=16)

        assert adc.data_path == adc_file
        assert adc.gain_config == 16
        assert adc.gain == 0.256  # Gain for config=16

    def test_adc_with_ascii_file(self, tmp_path):
        """Test ADC detects ASCII file format."""
        content = b"1000000 1024\n2000000 2048\n"
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(content)

        adc = ADC(tmp_path, logpath=None, gain_config=16)
        assert adc.is_ascii is True

    def test_adc_with_binary_file(self, tmp_path):
        """Test ADC detects binary file format."""
        data = struct.pack("<dqf", 1000.0, 100, 1.5)
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(data)

        adc = ADC(tmp_path, logpath=None, gain_config=16)
        assert adc.is_ascii is False

    def test_load_ascii_data(self, tmp_path):
        """Test loading ASCII ADC data."""
        content = b"1000000 1024\n2000000 2048\n"
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(content)

        adc = ADC(tmp_path, logpath=None, gain_config=16)
        adc.load_data()

        assert adc.data is not None
        assert isinstance(adc.data, pl.DataFrame)
        assert adc.data.shape[0] == 2

    def test_load_binary_data(self, tmp_path):
        """Test loading binary ADC data."""
        data = struct.pack("<dqf", 1000.0, 100, 1.5)
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(data)

        adc = ADC(tmp_path, logpath=None, gain_config=16)
        adc.load_data()

        assert adc.data is not None
        assert isinstance(adc.data, pl.DataFrame)
        assert adc.data.shape[0] == 1

    def test_plot_without_data_raises_error(self, tmp_path):
        """Test that plotting without loaded data raises ValueError."""
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(b"dummy")

        adc = ADC(tmp_path, logpath=None, gain_config=16)

        with pytest.raises(ValueError, match="No data loaded"):
            adc.plot()

    @patch("matplotlib.pyplot.show")
    def test_plot_with_data(self, mock_show, tmp_path):
        """Test plotting with loaded data."""
        content = b"1000000 1024\n2000000 2048\n"
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(content)

        adc = ADC(tmp_path, logpath=None, gain_config=16)
        adc.load_data()
        adc.plot()

        mock_show.assert_called_once()

    def test_auto_detect_gain_from_config(self, tmp_path):
        """Test auto-detection of gain from config.yml file."""
        # Create config file with ADC gain
        config_content = """
sensors:
  ADC_1:
    configuration:
      gain: 8
"""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(config_content)

        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(b"dummy")

        adc = ADC(tmp_path, logpath=None)
        assert adc.gain_config == 8
        assert adc.gain == 0.512

    def test_default_gain_when_no_config(self, tmp_path):
        """Test default gain=16 when no config file exists."""
        adc_file = tmp_path / "test_adc.bin"
        adc_file.write_bytes(b"dummy")

        adc = ADC(tmp_path, logpath=None)
        assert adc.gain_config == 16
        assert adc.gain == 0.256
