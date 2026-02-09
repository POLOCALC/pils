"""Tests for IMU sensor module."""

import polars as pl
import pytest

from pils.sensors.IMU import IMU, IMUSensor


class TestIMUSensor:
    """Test suite for IMUSensor class."""

    @pytest.fixture
    def sample_baro_csv(self, tmp_path):
        """Create sample barometer CSV file."""
        baro_file = tmp_path / "barometer.bin"
        # Write space-separated data
        baro_file.write_text(
            "1000000 101325.0 20.5\n2000000 101320.0 20.6\n3000000 101315.0 20.7\n"
        )
        return baro_file

    @pytest.fixture
    def sample_accelero_csv(self, tmp_path):
        """Create sample accelerometer CSV file."""
        accel_file = tmp_path / "accelerometer.bin"
        accel_file.write_text(
            "1000000 0.1 0.2 9.8\n2000000 0.15 0.25 9.85\n3000000 0.2 0.3 9.9\n"
        )
        return accel_file

    def test_init_with_string_path(self, sample_baro_csv):
        """Test IMUSensor initialization with string path."""
        sensor = IMUSensor(str(sample_baro_csv), "baro")
        assert sensor.path == str(sample_baro_csv)
        assert sensor.type == "baro"
        assert sensor.data is None

    def test_init_with_path_object(self, sample_baro_csv):
        """Test IMUSensor initialization with Path object."""
        sensor = IMUSensor(sample_baro_csv, "baro")
        assert sensor.path == sample_baro_csv
        assert sensor.type == "baro"
        assert sensor.data is None

    def test_load_data_creates_dataframe(self, sample_baro_csv):
        """Test load_data creates polars DataFrame."""
        sensor = IMUSensor(sample_baro_csv, "baro")
        sensor.load_data()
        assert isinstance(sensor.data, pl.DataFrame)
        assert sensor.data.shape[0] == 3

    def test_load_data_correct_columns(self, sample_baro_csv):
        """Test load_data creates correct columns for barometer."""
        sensor = IMUSensor(sample_baro_csv, "baro")
        sensor.load_data()
        expected_cols = ["timestamp", "pressure", "temperature", "datetime"]
        assert all(col in sensor.data.columns for col in expected_cols)

    def test_load_data_timestamp_conversion(self, sample_baro_csv):
        """Test timestamp conversion to datetime."""
        sensor = IMUSensor(sample_baro_csv, "baro")
        sensor.load_data()
        assert "datetime" in sensor.data.columns
        assert sensor.data["datetime"].dtype == pl.Datetime("us")

    def test_accelerometer_columns(self, sample_accelero_csv):
        """Test accelerometer has correct columns."""
        sensor = IMUSensor(sample_accelero_csv, "accelero")
        sensor.load_data()
        expected_cols = ["timestamp", "acc_x", "acc_y", "acc_z", "datetime"]
        assert all(col in sensor.data.columns for col in expected_cols)


class TestIMU:
    """Test suite for IMU class."""

    @pytest.fixture
    def imu_dir(self, tmp_path):
        """Create IMU directory with all sensor files."""
        imu_path = tmp_path / "imu"
        imu_path.mkdir()

        # Create barometer file
        (imu_path / "barometer.bin").write_text(
            "1000000 101325.0 20.5\n2000000 101320.0 20.6\n"
        )

        # Create accelerometer file
        (imu_path / "accelerometer.bin").write_text(
            "1000000 0.1 0.2 9.8\n2000000 0.15 0.25 9.85\n"
        )

        # Create gyroscope file
        (imu_path / "gyroscope.bin").write_text(
            "1000000 0.01 0.02 0.03\n2000000 0.015 0.025 0.035\n"
        )

        # Create magnetometer file
        (imu_path / "magnetometer.bin").write_text(
            "1000000 50.0 -20.0 30.0\n2000000 51.0 -21.0 31.0\n"
        )

        return imu_path

    def test_init_creates_all_sensors(self, imu_dir):
        """Test IMU initialization creates all four sensors."""
        imu = IMU(imu_dir)
        assert isinstance(imu.barometer, IMUSensor)
        assert isinstance(imu.accelerometer, IMUSensor)
        assert isinstance(imu.gyroscope, IMUSensor)
        assert isinstance(imu.magnetometer, IMUSensor)

    def test_init_with_path_type(self, imu_dir):
        """Test IMU accepts Path type."""
        imu = IMU(imu_dir)
        assert imu.dirpath == imu_dir

    def test_load_all_loads_all_sensors(self, imu_dir):
        """Test load_all() loads data for all sensors."""
        imu = IMU(imu_dir)
        imu.load_all()

        assert imu.barometer.data is not None
        assert imu.accelerometer.data is not None
        assert imu.gyroscope.data is not None
        assert imu.magnetometer.data is not None

    def test_load_all_correct_data_shapes(self, imu_dir):
        """Test load_all() creates correct shapes."""
        imu = IMU(imu_dir)
        imu.load_all()

        assert imu.barometer.data.shape[0] == 2
        assert imu.accelerometer.data.shape[0] == 2
        assert imu.gyroscope.data.shape[0] == 2
        assert imu.magnetometer.data.shape[0] == 2

    def test_sensor_data_types(self, imu_dir):
        """Test all sensors return polars DataFrames."""
        imu = IMU(imu_dir)
        imu.load_all()

        assert isinstance(imu.barometer.data, pl.DataFrame)
        assert isinstance(imu.accelerometer.data, pl.DataFrame)
        assert isinstance(imu.gyroscope.data, pl.DataFrame)
        assert isinstance(imu.magnetometer.data, pl.DataFrame)

    def test_barometer_has_pressure(self, imu_dir):
        """Test barometer data has pressure column."""
        imu = IMU(imu_dir)
        imu.load_all()
        assert "pressure" in imu.barometer.data.columns
        assert imu.barometer.data["pressure"][0] == 101325.0

    def test_accelerometer_has_acc_columns(self, imu_dir):
        """Test accelerometer has acc_x, acc_y, acc_z columns."""
        imu = IMU(imu_dir)
        imu.load_all()
        assert "acc_x" in imu.accelerometer.data.columns
        assert "acc_y" in imu.accelerometer.data.columns
        assert "acc_z" in imu.accelerometer.data.columns

    def test_gyroscope_has_xyz_columns(self, imu_dir):
        """Test gyroscope has x, y, z columns."""
        imu = IMU(imu_dir)
        imu.load_all()
        assert "x" in imu.gyroscope.data.columns
        assert "y" in imu.gyroscope.data.columns
        assert "z" in imu.gyroscope.data.columns

    def test_magnetometer_has_mag_columns(self, imu_dir):
        """Test magnetometer has mag_x, mag_y, mag_z columns."""
        imu = IMU(imu_dir)
        imu.load_all()
        assert "mag_x" in imu.magnetometer.data.columns
        assert "mag_y" in imu.magnetometer.data.columns
        assert "mag_z" in imu.magnetometer.data.columns
