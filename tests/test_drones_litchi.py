"""Test suite for Litchi drone module following TDD methodology."""

import polars as pl
import pytest

from pils.drones.litchi import Litchi


class TestLitchi:
    """Test suite for Litchi CSV flight log loading."""

    @pytest.fixture
    def sample_litchi_csv(self, tmp_path):
        """Create sample Litchi CSV file."""
        csv_content = """latitude,longitude,altitude(m),speed(mps),distance(m),velocityX(mps),velocityY(mps),velocityZ(mps),pitch(deg),roll(deg),yaw(deg),batteryTemperature,pitchRaw,rollRaw,yawRaw,gimbalPitchRaw,gimbalRollRaw,gimbalYawRaw,datetime(utc),isflying
40.7128,-74.0060,100.5,5.2,0.0,2.5,3.1,0.5,-5.0,2.0,180.0,25.5,-5.0,2.0,180.0,-15.0,0.0,180.0,2024-01-15 10:30:00.123,1
40.7129,-74.0061,101.0,5.3,10.5,2.6,3.2,0.6,-5.1,2.1,181.0,25.6,-5.1,2.1,181.0,-15.1,0.1,181.0,2024-01-15 10:30:01.123,1
40.7130,-74.0062,101.5,5.4,21.0,2.7,3.3,0.7,-5.2,2.2,182.0,25.7,-5.2,2.2,182.0,-15.2,0.2,182.0,2024-01-15 10:30:02.123,1"""

        csv_path = tmp_path / "litchi_flight.csv"
        csv_path.write_text(csv_content)
        return csv_path

    @pytest.fixture
    def minimal_litchi_csv(self, tmp_path):
        """Create minimal Litchi CSV with required columns."""
        csv_content = """latitude,longitude,datetime(utc)
40.7128,-74.0060,2024-01-15 10:30:00
40.7129,-74.0061,2024-01-15 10:30:01"""

        csv_path = tmp_path / "minimal.csv"
        csv_path.write_text(csv_content)
        return csv_path

    def test_init_with_string_path(self, sample_litchi_csv):
        """Test Litchi initialization with string path."""
        litchi = Litchi(str(sample_litchi_csv))
        assert litchi.path == str(sample_litchi_csv)
        assert litchi.data is None

    def test_init_with_path_object(self, sample_litchi_csv):
        """Test Litchi initialization with Path object."""
        litchi = Litchi(sample_litchi_csv)
        assert litchi.path == sample_litchi_csv
        assert litchi.data is None

    def test_load_data_success(self, sample_litchi_csv):
        """Test successful loading of Litchi CSV."""
        litchi = Litchi(sample_litchi_csv)
        litchi.load_data()
        assert litchi.data is not None
        assert isinstance(litchi.data, pl.DataFrame)
        assert litchi.data.shape[0] == 3

    def test_load_data_datetime_conversion(self, sample_litchi_csv):
        """Test that datetime column is properly converted."""
        litchi = Litchi(sample_litchi_csv)
        litchi.load_data()
        assert "datetime" in litchi.data.columns
        assert "datetime(utc)" not in litchi.data.columns
        # Check datetime type
        assert litchi.data["datetime"].dtype == pl.Datetime

    def test_load_data_with_column_selection(self, sample_litchi_csv):
        """Test loading with specific column selection."""
        litchi = Litchi(sample_litchi_csv)
        cols = ["latitude", "longitude", "altitude(m)", "datetime(utc)"]
        litchi.load_data(cols=cols)
        assert litchi.data is not None
        # Should have selected columns (minus datetime(utc), plus datetime)
        assert "latitude" in litchi.data.columns
        assert "longitude" in litchi.data.columns
        assert "datetime" in litchi.data.columns

    def test_load_data_drops_nan_zero_cols(self, tmp_path):
        """Test that NaN and zero columns are dropped."""
        csv_content = """latitude,longitude,datetime(utc),allzero,allnan
40.7128,-74.0060,2024-01-15 10:30:00,0,
40.7129,-74.0061,2024-01-15 10:30:01,0,"""

        csv_path = tmp_path / "with_empty.csv"
        csv_path.write_text(csv_content)

        litchi = Litchi(csv_path)
        litchi.load_data(
            cols=["latitude", "longitude", "datetime(utc)", "allzero", "allnan"]
        )
        # Columns with all zeros or NaN should be dropped
        assert "allzero" not in litchi.data.columns or litchi.data.shape[1] < 5

    def test_load_data_default_columns(self, sample_litchi_csv):
        """Test loading with default column list."""
        litchi = Litchi(sample_litchi_csv)
        litchi.load_data()  # Use default cols
        # Check that common columns are present
        assert "latitude" in litchi.data.columns
        assert "longitude" in litchi.data.columns
        assert "datetime" in litchi.data.columns


class TestLitchiInit:
    """Test suite for Litchi initialization."""

    def test_init_attributes(self, tmp_path):
        """Test that all attributes are initialized correctly."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text(
            "latitude,longitude,datetime(utc)\n40.0,-74.0,2024-01-15 10:00:00"
        )

        litchi = Litchi(csv_path)
        assert litchi.path == csv_path
        assert litchi.data is None

    def test_init_with_nonexistent_path(self):
        """Test initialization with non-existent path."""
        # Should not raise error on init, only on load_data
        litchi = Litchi("nonexistent.csv")
        assert litchi.path == "nonexistent.csv"
        assert litchi.data is None

    def test_load_data_file_not_found(self):
        """Test that load_data raises error for non-existent file."""
        litchi = Litchi("nonexistent.csv")
        with pytest.raises(
            (ValueError, FileNotFoundError)
        ):  # pl.read_csv will raise an error
            litchi.load_data()
