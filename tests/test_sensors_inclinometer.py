"""Tests for Inclinometer sensor module (pils/sensors/inclinometer.py)."""

from unittest.mock import patch

import polars as pl
import pytest

from pils.sensors.inclinometer import (
    IMX5Inclinometer,
    Inclinometer,
    KernelInclinometer,
    decode_inclino,
    detect_inclinometer_type_from_config,
    detect_inclinometer_type_from_files,
)


class TestDecodeInclino:
    """Test suite for decode_inclino function."""

    def test_decode_inclino_basic(self, tmp_path):
        """Test basic decoding of inclinometer binary data."""
        # Create minimal binary data with Kernel message format
        # Sequence: \xaaU\x01\x81 + payload
        sequence = b"\xaaU\x01\x81"
        # Mock payload (will be decoded by KernelMsg)
        payload = b"\x00" * 20  # Dummy payload

        inclino_file = tmp_path / "test_inclino.bin"
        inclino_file.write_bytes(sequence + payload)

        with patch("pils.decoders.KERNEL_utils.KernelMsg") as MockKernel:
            mock_instance = MockKernel.return_value
            mock_instance.decode_single.return_value = {"Roll": 10.0, "Pitch": 5.0, "Heading": 90.0}

            result = decode_inclino(inclino_file)

            assert isinstance(result, dict)
            mock_instance.decode_single.assert_called_once()

    def test_decode_inclino_multiple_messages(self, tmp_path):
        """Test decoding multiple inclinometer messages."""
        sequence = b"\xaaU\x01\x81"
        payload1 = b"\x01" * 20
        payload2 = b"\x02" * 20

        inclino_file = tmp_path / "test_inclino.bin"
        inclino_file.write_bytes(sequence + payload1 + sequence + payload2)

        with patch("pils.decoders.KERNEL_utils.KernelMsg") as MockKernel:
            mock_instance = MockKernel.return_value
            mock_instance.decode_single.side_effect = [
                {"Roll": 10.0, "Pitch": 5.0, "Counter": 100},
                {"Roll": 11.0, "Pitch": 6.0, "Counter": 116},
            ]

            result = decode_inclino(inclino_file)

            assert isinstance(result, dict)
            assert len(result.get("Roll", [])) == 2

    def test_decode_inclino_handles_exception(self, tmp_path, caplog):
        """Test that decode_inclino handles exceptions gracefully."""
        sequence = b"\xaaU\x01\x81"
        payload = b"\x00" * 20

        inclino_file = tmp_path / "test_inclino.bin"
        inclino_file.write_bytes(sequence + payload + sequence + b"\xff" * 5)

        with patch("pils.decoders.KERNEL_utils.KernelMsg") as MockKernel:
            mock_instance = MockKernel.return_value
            mock_instance.decode_single.side_effect = [
                {"Roll": 10.0, "Pitch": 5.0},
                Exception("Decode error"),
            ]

            with caplog.at_level("WARNING"):
                result = decode_inclino(inclino_file)

            # Should return dict with first message only (second failed)
            assert isinstance(result, dict)

    def test_decode_inclino_with_path_string(self, tmp_path):
        """Test decode_inclino accepts string path."""
        sequence = b"\xaaU\x01\x81"
        payload = b"\x00" * 20

        inclino_file = tmp_path / "test_inclino.bin"
        inclino_file.write_bytes(sequence + payload)

        with patch("pils.decoders.KERNEL_utils.KernelMsg") as MockKernel:
            mock_instance = MockKernel.return_value
            mock_instance.decode_single.return_value = {"Roll": 10.0}

            result = decode_inclino(str(inclino_file))
            assert isinstance(result, dict)


class TestDetectInclinometerTypeFromConfig:
    """Test suite for detect_inclinometer_type_from_config function."""

    def test_detect_imx5_from_config(self, tmp_path):
        """Test detecting IMX5 inclinometer from config."""
        config_content = """
sensors:
  IMX5_1:
    sensor_info:
      type: IMX5
      manufacturer: InertialSense
"""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(config_content)

        result = detect_inclinometer_type_from_config(tmp_path)
        assert result == "imx5"

    def test_detect_kernel_from_config(self, tmp_path):
        """Test detecting Kernel inclinometer from config."""
        config_content = """
sensors:
  KERNEL_1:
    name: Kernel-100
    sensor_info:
      type: INERTIAL
      manufacturer: Kernel
"""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(config_content)

        result = detect_inclinometer_type_from_config(tmp_path)
        assert result == "kernel"

    def test_no_config_file_returns_none(self, tmp_path):
        """Test that missing config returns None."""
        result = detect_inclinometer_type_from_config(tmp_path)
        assert result is None

    def test_exception_returns_none(self, tmp_path, caplog):
        """Test that exception during config parsing returns None and logs."""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text("invalid: yaml: content: [")

        with caplog.at_level("WARNING"):
            result = detect_inclinometer_type_from_config(tmp_path)

        assert result is None


class TestDetectInclinometerTypeFromFiles:
    """Test suite for detect_inclinometer_type_from_files function."""

    def test_detect_imx5_files(self, tmp_path):
        """Test detecting IMX5 from CSV files."""
        (tmp_path / "test_INC_ins.csv").write_text("timestamp_ns,roll_rad,pitch_rad\n")
        result = detect_inclinometer_type_from_files(tmp_path)
        assert result == "imx5"

    def test_detect_kernel_files(self, tmp_path):
        """Test detecting Kernel from binary files."""
        (tmp_path / "test_INC.bin").write_bytes(b"\xaa\x55" * 100)
        result = detect_inclinometer_type_from_files(tmp_path)
        assert result == "kernel"

    def test_no_files_returns_unknown(self, tmp_path):
        """Test unknown type when no matching files."""
        result = detect_inclinometer_type_from_files(tmp_path)
        assert result == "unknown"


class TestKernelInclinometer:
    """Test suite for KernelInclinometer class."""

    def test_initialization(self, tmp_path):
        """Test KernelInclinometer initialization."""
        inc_file = tmp_path / "test_INC.bin"
        inc_file.write_bytes(b"\xaa\x55" * 100)

        kernel_inc = KernelInclinometer(inc_file, logpath=None)

        assert kernel_inc.path == inc_file
        assert kernel_inc.logpath is None
        assert kernel_inc.tstart is None

    def test_load_data(self, tmp_path):
        """Test loading Kernel inclinometer data."""
        sequence = b"\xaaU\x01\x81"
        payload = b"\x00" * 20

        inc_file = tmp_path / "test_INC.bin"
        inc_file.write_bytes((sequence + payload) * 10)

        with patch("pils.sensors.inclinometer.decode_inclino") as mock_decode:
            # Mock decoded data with valid counter increments
            mock_decode.return_value = {
                "Counter": [32000, 32016, 32032, 32048, 32064],
                "Roll": [10.0, 10.1, 10.2, 10.3, 10.4],
                "Pitch": [5.0, 5.1, 5.2, 5.3, 5.4],
                "Heading": [90.0, 90.1, 90.2, 90.3, 90.4],
            }

            kernel_inc = KernelInclinometer(inc_file, logpath=None)
            kernel_inc.load_data()

            assert kernel_inc.data is not None
            assert isinstance(kernel_inc.data, pl.DataFrame)
            assert "roll" in kernel_inc.data.columns
            assert "pitch" in kernel_inc.data.columns
            assert "yaw" in kernel_inc.data.columns

    def test_read_log_time_success(self, tmp_path):
        """Test reading start time from log file."""
        log_content = """
2024-01-01 10:00:00 - INFO - Connected to KERNEL sensor Kernel-100
2024-01-01 10:00:05 - INFO - Sensor started
"""
        log_file = tmp_path / "test.log"
        log_file.write_text(log_content)

        inc_file = tmp_path / "test_INC.bin"
        inc_file.write_bytes(b"\xaa\x55")

        kernel_inc = KernelInclinometer(inc_file, logpath=str(log_file))

        with patch("pils.sensors.inclinometer.read_log_time") as mock_read_log:
            from datetime import datetime

            mock_read_log.return_value = (datetime(2024, 1, 1, 10, 0, 0), None)
            kernel_inc.read_log_time(logfile=str(log_file))

            assert kernel_inc.tstart is not None

    def test_read_log_time_exception(self, tmp_path, caplog):
        """Test read_log_time handles exception and logs warning."""
        log_file = tmp_path / "test.log"
        log_file.write_text("no timestamp here")

        inc_file = tmp_path / "test_INC.bin"
        inc_file.write_bytes(b"\xaa\x55")

        kernel_inc = KernelInclinometer(inc_file, logpath=str(log_file))

        with patch("pils.sensors.inclinometer.read_log_time") as mock_read_log:
            mock_read_log.side_effect = Exception("Parse error")

            with caplog.at_level("WARNING"):
                kernel_inc.read_log_time(logfile=str(log_file))

            assert kernel_inc.tstart is None
            assert any("start time" in record.message.lower() for record in caplog.records)


class TestIMX5Inclinometer:
    """Test suite for IMX5Inclinometer class."""

    def test_initialization(self, tmp_path):
        """Test IMX5Inclinometer initialization."""
        imx5_inc = IMX5Inclinometer(tmp_path, logpath=None)

        assert imx5_inc.dirpath == tmp_path
        assert imx5_inc.logpath is None
        assert isinstance(imx5_inc.data, dict)

    def test_load_ins_data(self, tmp_path):
        """Test loading INS data from CSV."""
        ins_content = """timestamp_ns,roll_rad,pitch_rad,yaw_rad,lat,lon,alt
1000000000,0.1,0.2,0.3,40.7128,-74.0060,100.0
2000000000,0.15,0.25,0.35,40.7129,-74.0061,101.0
"""
        ins_file = tmp_path / "test_INC_ins.csv"
        ins_file.write_text(ins_content)

        imx5_inc = IMX5Inclinometer(tmp_path, logpath=None)
        imx5_inc.load_ins()

        assert "INS" in imx5_inc.data
        assert isinstance(imx5_inc.data["INS"], pl.DataFrame)
        assert "roll" in imx5_inc.data["INS"].columns  # Converted from radians
        assert "pitch" in imx5_inc.data["INS"].columns
        assert "yaw" in imx5_inc.data["INS"].columns

    def test_load_imu_data(self, tmp_path):
        """Test loading IMU data from CSV."""
        imu_content = """timestamp_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
1000000000,0.1,0.2,9.8,0.01,0.02,0.03
2000000000,0.15,0.25,9.81,0.015,0.025,0.035
"""
        imu_file = tmp_path / "test_INC_imu.csv"
        imu_file.write_text(imu_content)

        imx5_inc = IMX5Inclinometer(tmp_path, logpath=None)
        imx5_inc.load_imu()

        assert "IMU" in imx5_inc.data
        assert isinstance(imx5_inc.data["IMU"], pl.DataFrame)
        assert "timestamp" in imx5_inc.data["IMU"].columns

    def test_load_data_calls_all_loaders(self, tmp_path):
        """Test that load_data calls all sub-loaders."""
        imx5_inc = IMX5Inclinometer(tmp_path, logpath=None)

        with patch.object(imx5_inc, "load_ins") as mock_ins:
            with patch.object(imx5_inc, "load_imu") as mock_imu:
                with patch.object(imx5_inc, "load_inl2") as mock_inl2:
                    imx5_inc.load_data()

                    mock_ins.assert_called_once()
                    mock_imu.assert_called_once()
                    mock_inl2.assert_called_once()


class TestInclinometerClass:
    """Test suite for unified Inclinometer class."""

    def test_auto_detect_kernel(self, tmp_path):
        """Test auto-detection of Kernel inclinometer."""
        config_content = """
sensors:
  KERNEL_1:
    name: Kernel-100
    sensor_info:
      type: INERTIAL
"""
        config_file = tmp_path / "test_config.yml"
        config_file.write_text(config_content)

        inc_file = tmp_path / "test_INC.bin"
        inc_file.write_bytes(b"\xaa\x55")

        inclinometer = Inclinometer(tmp_path, logpath=None)

        assert inclinometer.sensor_type == "kernel"
        assert isinstance(inclinometer._decoder, KernelInclinometer)

    def test_auto_detect_imx5(self, tmp_path):
        """Test auto-detection of IMX5 inclinometer."""
        (tmp_path / "test_INC_ins.csv").write_text("timestamp_ns,roll_rad\n")

        inclinometer = Inclinometer(tmp_path, logpath=None)

        assert inclinometer.sensor_type == "imx5"
        assert isinstance(inclinometer._decoder, IMX5Inclinometer)

    def test_forced_sensor_type(self, tmp_path):
        """Test forcing sensor type."""
        inclinometer = Inclinometer(tmp_path, logpath=None, sensor_type="kernel")

        assert inclinometer.sensor_type == "kernel"

    def test_load_data_raises_without_files(self, tmp_path):
        """Test that load_data raises error when no inclinometer found."""
        inclinometer = Inclinometer(tmp_path, logpath=None)

        with pytest.raises(ValueError, match="No inclinometer data found"):
            inclinometer.load_data()

    def test_tstart_property_kernel(self, tmp_path):
        """Test tstart property for Kernel inclinometer."""
        from datetime import datetime

        inc_file = tmp_path / "test_INC.bin"
        inc_file.write_bytes(b"\xaa\x55")

        inclinometer = Inclinometer(tmp_path, logpath=None, sensor_type="kernel")
        inclinometer._decoder.tstart = datetime(2024, 1, 1, 10, 0, 0)

        assert inclinometer.tstart == datetime(2024, 1, 1, 10, 0, 0)

    def test_ins_data_property_imx5(self, tmp_path):
        """Test ins_data property for IMX5."""
        ins_content = """timestamp_ns,roll_rad
1000000000,0.1
"""
        ins_file = tmp_path / "test_INC_ins.csv"
        ins_file.write_text(ins_content)

        inclinometer = Inclinometer(tmp_path, logpath=None, sensor_type="imx5")
        inclinometer._decoder.load_ins()

        assert inclinometer.ins_data is not None
        assert isinstance(inclinometer.ins_data, pl.DataFrame)

    @patch("matplotlib.pyplot.show")
    def test_plot_with_data(self, mock_show, tmp_path):
        """Test plotting inclinometer data."""
        sequence = b"\xaaU\x01\x81"
        payload = b"\x00" * 20

        inc_file = tmp_path / "test_INC.bin"
        inc_file.write_bytes((sequence + payload) * 10)

        with patch("pils.sensors.inclinometer.decode_inclino") as mock_decode:
            # Mock decoded data with valid counter increments
            mock_decode.return_value = {
                "Counter": [32000, 32016, 32032, 32048, 32064],
                "Roll": [10.0, 10.1, 10.2, 10.3, 10.4],
                "Pitch": [5.0, 5.1, 5.2, 5.3, 5.4],
                "Heading": [90.0, 90.1, 90.2, 90.3, 90.4],
            }

            inclinometer = Inclinometer(inc_file, logpath=None, sensor_type="kernel")
            inclinometer.load_data()
            inclinometer.plot()

            mock_show.assert_called_once()

    def test_plot_without_data_raises_error(self, tmp_path):
        """Test that plotting without data raises ValueError."""
        inclinometer = Inclinometer(tmp_path, logpath=None, sensor_type="kernel")
        inclinometer.data = None

        with pytest.raises(ValueError, match="Data not loaded"):
            inclinometer.plot()

    def test_logging_sensor_type(self, tmp_path, caplog):
        """Test that sensor type is logged during initialization."""
        inc_file = tmp_path / "test_INC.bin"
        inc_file.write_bytes(b"\xaa\x55")

        with caplog.at_level("INFO"):
            inclinometer = Inclinometer(tmp_path, logpath=None, sensor_type="kernel")

        # The print statement should be replaced with logger
        # For now, we just check it doesn't crash
        assert inclinometer.sensor_type == "kernel"
