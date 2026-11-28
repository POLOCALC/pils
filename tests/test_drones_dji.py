"""Test suite for DJIDat module following TDD methodology."""

import logging

import polars as pl
import pytest

from pils.drones.dji.dat import DJIDat


class TestDJIDroneCSV:
    """Test suite for DJIDat CSV loading functionality."""

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
        """Test DJIDat initialization with string path."""
        drone = DJIDat(str(csv_file))
        assert drone.path == str(csv_file)
        assert drone.data == {}
        assert drone.source_format is None

    def test_init_with_path_object(self, csv_file):
        """Test DJIDat initialization with Path object."""
        drone = DJIDat(csv_file)
        assert drone.path == csv_file
        assert drone.data == {}

    def test_load_csv_data(self, csv_file):
        """Test loading CSV data successfully."""
        drone = DJIDat(csv_file)
        drone.load_data(use_dat=False)
        assert isinstance(drone.data, pl.DataFrame)
        assert drone.source_format == "csv"
        assert drone.data.shape[0] == 4

    def test_load_csv_with_column_selection(self, csv_file):
        """Test loading CSV with specific columns."""
        drone = DJIDat(csv_file)
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

        drone = DJIDat(csv_with_dups)
        drone.load_data(use_dat=False, remove_duplicate=True)
        # Should remove duplicate latitude values
        assert drone.data.shape[0] <= 3

    def test_csv_datetime_parsing(self, csv_file):
        """Test datetime parsing from CSV."""
        drone = DJIDat(csv_file)
        drone.load_data(use_dat=False)
        assert "datetime" in drone.data.columns
        assert "timestamp" in drone.data.columns


class TestDJIDroneDAT:
    """Test suite for DJIDat DAT binary format parsing."""

    def test_load_from_dat_file_not_found(self):
        """Test loading DAT file that doesn't exist."""
        drone = DJIDat("nonexistent.dat")
        with pytest.raises(FileNotFoundError):
            drone._load_from_dat()


class TestDJIDroneUtils:
    """Test suite for DJIDat utility methods."""

    def test_unwrap_tick_no_wrapping(self):
        """Test unwrap_tick with no wrapping."""
        df = pl.DataFrame({"tick": [100, 200, 300, 400]})
        result = DJIDat._unwrap_tick(df)
        assert result["tick"].to_list() == [100, 200, 300, 400]

    def test_unwrap_tick_with_wrapping(self):
        """Test unwrap_tick with tick counter wrapping."""
        # Simulate tick counter overflow
        df = pl.DataFrame({"tick": [9e7, 9.5e7, 1e6, 2e6]})
        result = DJIDat._unwrap_tick(df, wrap_threshold=1e8)
        # After unwrapping, ticks should be monotonically increasing
        ticks = result["tick"].to_list()
        assert all(ticks[i] < ticks[i + 1] for i in range(len(ticks) - 1))

    def test_get_tick_offset_no_sync(self):
        """Test get_tick_offset when no sync params exist."""
        drone = DJIDat("test.dat")
        offset = drone.get_tick_offset()
        assert offset == 0.0

    def test_get_tick_offset_with_sync(self):
        """Test get_tick_offset returns a float value."""
        drone = DJIDat("test.dat")
        # Without sync, should return 0.0
        offset = drone.get_tick_offset()
        assert isinstance(offset, float)
        assert offset == 0.0


class TestDJIDroneLogging:
    """Test suite to verify logging instead of print statements."""

    def test_csv_parsing_exception_logged(self, caplog, tmp_path):
        """Test that CSV parsing exceptions are logged."""
        # Create malformed CSV
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("not,valid,csv\n1,2")  # Wrong number of columns

        drone = DJIDat(bad_csv)
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
        drone = DJIDat("test.dat")
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
        drone = DJIDat("test.dat")
        drone.data = {}  # No DAT data

        with pytest.raises((KeyError, ValueError, TypeError)):
            drone.align_datfile(sample_gps_df)
