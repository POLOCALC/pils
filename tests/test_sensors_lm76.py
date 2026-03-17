"""Tests for LM76 temperature sensor module (pils/sensors/LM76.py)."""

from pathlib import Path

import polars as pl
import pytest

from pils.sensors.LM76 import LM76


def _make_tmp_csv(directory: Path) -> Path:
    """Create a sample TMP CSV file with realistic LM76 data."""
    content = (
        "timestamp_ns,temperature_c,status_crit,status_high,status_low\n"
        "1000000000,25.5,0,0,0\n"
        "2000000000,26.0,0,0,0\n"
        "3000000000,26.5,0,1,0\n"
    )
    csv_file = directory / "sensor_TMP_data.csv"
    csv_file.write_text(content)
    return csv_file


class TestLM76Init:
    """Test suite for LM76 __init__ method."""

    def test_init_finds_tmp_csv(self, tmp_path):
        """Test that __init__ finds TMP CSV file in path."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        csv_file = _make_tmp_csv(sensor_dir)

        lm76 = LM76(sensor_dir)

        assert lm76.data_path == csv_file

    def test_init_no_tmp_csv_sets_data_path_none(self, tmp_path):
        """Test that data_path is None when no TMP CSV file exists."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()

        lm76 = LM76(sensor_dir)

        assert lm76.data_path is None

    def test_init_path_attribute(self, tmp_path):
        """Test that path attribute is set correctly."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()

        lm76 = LM76(sensor_dir)

        assert lm76.path == sensor_dir

    def test_init_data_is_none(self, tmp_path):
        """Test that data is None after initialization."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()

        lm76 = LM76(sensor_dir)

        assert lm76.data is None

    def test_init_with_explicit_logpath(self, tmp_path):
        """Test that explicit logpath is stored as-is."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        log_file = tmp_path / "flight.log"
        log_file.write_text("log content")

        lm76 = LM76(sensor_dir, logpath=log_file)

        assert lm76.logpath == log_file

    def test_init_logpath_none_when_no_data_file(self, tmp_path):
        """Test that logpath is None when data_path is None and inference fails."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        # No TMP CSV file → get_logpath_from_datapath raises FileNotFoundError

        lm76 = LM76(sensor_dir)

        assert lm76.logpath is None

    def test_init_accepts_path_object(self, tmp_path):
        """Test that __init__ accepts a Path object."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()

        lm76 = LM76(Path(sensor_dir))

        assert isinstance(lm76.path, Path)


class TestLM76ReadSettingsFromConfig:
    """Test suite for LM76._read_settings_from_config method."""

    def test_read_settings_returns_lm76_configuration(self, tmp_path):
        """Test that settings are read from *_config.yml in the parent directory."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        # Config file is expected in os.path.dirname(self.path) = tmp_path
        config_content = """
sensors:
  LM76_1:
    configuration:
      interval: 1000
      tcrit: 80
      thyst: 75
      tlow: 10
      thigh: 70
"""
        (tmp_path / "flight_config.yml").write_text(config_content)

        lm76 = LM76(sensor_dir)
        settings = lm76._read_settings_from_config()

        assert settings is not None
        assert settings["interval"] == 1000
        assert settings["tcrit"] == 80
        assert settings["thyst"] == 75
        assert settings["tlow"] == 10
        assert settings["thigh"] == 70

    def test_read_settings_no_config_returns_none(self, tmp_path):
        """Test that None is returned when no config file exists."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        lm76 = LM76(sensor_dir)
        settings = lm76._read_settings_from_config()

        assert settings is None

    def test_read_settings_no_lm76_sensor_returns_none(self, tmp_path):
        """Test that None is returned when config has no sensor key starting with LM76."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        config_content = """
sensors:
  ADC_1:
    configuration:
      gain: 16
"""
        (tmp_path / "flight_config.yml").write_text(config_content)

        lm76 = LM76(sensor_dir)
        settings = lm76._read_settings_from_config()

        assert settings is None

    def test_read_settings_invalid_yaml_returns_none(self, tmp_path):
        """Test that invalid YAML config returns None without raising."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        (tmp_path / "flight_config.yml").write_text("invalid: yaml: content: [[[")

        lm76 = LM76(sensor_dir)
        settings = lm76._read_settings_from_config()

        assert settings is None

    def test_read_settings_multiple_sensors_picks_lm76(self, tmp_path):
        """Test that only the LM76 sensor section is returned."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        config_content = """
sensors:
  ADC_1:
    configuration:
      gain: 16
  LM76_1:
    configuration:
      interval: 500
      tcrit: 85
  GPS_1:
    configuration:
      baud: 115200
"""
        (tmp_path / "flight_config.yml").write_text(config_content)

        lm76 = LM76(sensor_dir)
        settings = lm76._read_settings_from_config()

        assert settings is not None
        assert settings["interval"] == 500


class TestLM76LoadData:
    """Test suite for LM76.load_data method."""

    def test_load_data_populates_data(self, tmp_path):
        """Test that load_data populates self.data."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        lm76 = LM76(sensor_dir)
        lm76.load_data()

        assert lm76.data is not None

    def test_load_data_returns_dict_with_expected_keys(self, tmp_path):
        """Test that self.data is a dict with 'data' and 'metadata' keys."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        lm76 = LM76(sensor_dir)
        lm76.load_data()

        assert isinstance(lm76.data, dict)
        assert "data" in lm76.data
        assert "metadata" in lm76.data

    def test_load_data_stores_polars_dataframe(self, tmp_path):
        """Test that data['data'] is a Polars DataFrame."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        lm76 = LM76(sensor_dir)
        lm76.load_data()

        assert isinstance(lm76.data["data"], pl.DataFrame)

    def test_load_data_adds_timestamp_column(self, tmp_path):
        """Test that load_data adds a 'timestamp' column in seconds."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        lm76 = LM76(sensor_dir)
        lm76.load_data()

        df = lm76.data["data"]
        assert "timestamp" in df.columns
        assert df["timestamp"][0] == pytest.approx(1.0, abs=1e-6)
        assert df["timestamp"][1] == pytest.approx(2.0, abs=1e-6)
        assert df["timestamp"][2] == pytest.approx(3.0, abs=1e-6)

    def test_load_data_preserves_temperature_column(self, tmp_path):
        """Test that temperature_c column is preserved with correct values."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        lm76 = LM76(sensor_dir)
        lm76.load_data()

        df = lm76.data["data"]
        assert "temperature_c" in df.columns
        assert df["temperature_c"][0] == pytest.approx(25.5, abs=1e-6)

    def test_load_data_correct_row_count(self, tmp_path):
        """Test that load_data loads all rows from CSV."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        lm76 = LM76(sensor_dir)
        lm76.load_data()

        assert lm76.data["data"].shape[0] == 3

    def test_load_data_metadata_none_without_config(self, tmp_path):
        """Test that metadata is None when no config file is present."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        lm76 = LM76(sensor_dir)
        lm76.load_data()

        assert lm76.data["metadata"] is None

    def test_load_data_metadata_populated_with_config(self, tmp_path):
        """Test that metadata is populated from config file."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()
        _make_tmp_csv(sensor_dir)

        config_content = """
sensors:
  LM76_1:
    configuration:
      interval: 500
      tcrit: 85
"""
        (tmp_path / "flight_config.yml").write_text(config_content)

        lm76 = LM76(sensor_dir)
        lm76.load_data()

        assert lm76.data["metadata"] is not None
        assert lm76.data["metadata"]["interval"] == 500

    def test_load_data_raises_file_not_found(self, tmp_path):
        """Test that load_data raises FileNotFoundError when the CSV does not exist."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()

        lm76 = LM76(sensor_dir)
        lm76.data_path = sensor_dir / "nonexistent_TMP_file.csv"

        with pytest.raises(FileNotFoundError):
            lm76.load_data()

    @pytest.mark.parametrize(
        "timestamp_ns,expected_s",
        [
            (1_000_000_000, 1.0),
            (500_000_000, 0.5),
            (10_000_000_000, 10.0),
        ],
    )
    def test_load_data_timestamp_conversion(self, tmp_path, timestamp_ns, expected_s):
        """Test that timestamp_ns is correctly converted to seconds."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()

        content = f"timestamp_ns,temperature_c,status_crit,status_high,status_low\n{timestamp_ns},25.0,0,0,0\n"
        (sensor_dir / "sensor_TMP_data.csv").write_text(content)

        lm76 = LM76(sensor_dir)
        lm76.load_data()

        assert lm76.data["data"]["timestamp"][0] == pytest.approx(expected_s, abs=1e-9)


class TestLM76Plot:
    """Test suite for LM76.plot method."""

    def test_plot_raises_value_error_without_data(self, tmp_path):
        """Test that plot raises ValueError when data has not been loaded."""
        sensor_dir = tmp_path / "lm76_data"
        sensor_dir.mkdir()

        lm76 = LM76(sensor_dir)

        with pytest.raises(ValueError, match="No data loaded"):
            lm76.plot()
