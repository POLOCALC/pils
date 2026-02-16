"""Comprehensive tests for Camera sensor module - all three modes."""

from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from pils.sensors.camera import Camera


class TestCameraPhotogrammetryMode:
    """Test suite for photogrammetry mode."""

    @pytest.fixture
    def photogrammetry_csv(self, tmp_path):
        """Create sample photogrammetry CSV file."""
        csv_path = tmp_path / "photogrammetry.csv"
        df = pl.DataFrame(
            {
                "timestamp": [1000.0, 2000.0, 3000.0, 4000.0],
                "pitch": [0.1, 0.2, 0.3, 0.4],
                "roll": [0.05, 0.15, 0.25, 0.35],
                "yaw": [1.5, 1.6, 1.7, 1.8],
            }
        )
        df.write_csv(csv_path)
        return csv_path

    def test_photogrammetry_mode_loads_csv(self, photogrammetry_csv):
        """Test loading photogrammetry CSV data."""
        camera = Camera(photogrammetry_csv, use_photogrammetry=True)
        camera.load_data()

        assert camera.data is not None
        df, model = camera.data

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 4
        assert model is None

    def test_photogrammetry_validates_columns(self, photogrammetry_csv):
        """Test that required columns are present."""
        camera = Camera(photogrammetry_csv, use_photogrammetry=True)
        camera.load_data()

        df, _ = camera.data
        assert "timestamp" in df.columns
        assert "pitch" in df.columns
        assert "roll" in df.columns
        assert "yaw" in df.columns

    def test_photogrammetry_returns_none_camera_model(self, photogrammetry_csv):
        """Test that photogrammetry mode returns None for camera model."""
        camera = Camera(photogrammetry_csv, use_photogrammetry=True)
        camera.load_data()

        _, model = camera.data
        assert model is None

    def test_photogrammetry_data_structure(self, photogrammetry_csv):
        """Test DataFrame schema for photogrammetry mode."""
        camera = Camera(photogrammetry_csv, use_photogrammetry=True)
        camera.load_data()

        df, _ = camera.data

        # Check data types
        assert df["timestamp"].dtype == pl.Float64
        assert df["pitch"].dtype == pl.Float64
        assert df["roll"].dtype == pl.Float64
        assert df["yaw"].dtype == pl.Float64

        # Check values
        assert df["timestamp"][0] == 1000.0
        assert df["pitch"][0] == 0.1


class TestCameraSonyMode:
    """Test suite for Sony RX0 MarkII camera mode."""

    @pytest.fixture
    def sony_video_and_log(self, tmp_path):
        """Create Sony video file and log file."""
        # Create camera directory
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        # Create video file
        video_file = camera_dir / "video.mp4"
        video_file.write_bytes(b"fake_sony_video")

        # Create log file in parent directory
        log_file = tmp_path / "session.log"
        log_content = "2024/11/20 14:30:45.123000 [INFO] Camera Sony starts recording\n"
        log_file.write_text(log_content)

        return camera_dir, video_file, log_file

    @pytest.fixture
    def mock_telemetry_parser(self):
        """Mock telemetry_parser.Parser with normalized IMU data."""
        with patch("pils.sensors.camera.telemetry_parser") as mock_parser_module:
            # Create mock parser instance
            mock_parser = Mock()

            # Mock normalized_imu() with realistic data
            mock_parser.normalized_imu.return_value = [
                {
                    "timestamp_ms": 0,
                    "gyro": [0.1, 0.2, 0.3],
                    "accl": [0.0, 0.0, 9.8],
                },
                {
                    "timestamp_ms": 100,
                    "gyro": [0.15, 0.25, 0.35],
                    "accl": [0.1, 0.1, 9.7],
                },
                {
                    "timestamp_ms": 200,
                    "gyro": [0.2, 0.3, 0.4],
                    "accl": [0.2, 0.2, 9.6],
                },
            ]

            # Set up Parser constructor to return mock instance
            mock_parser_module.Parser.return_value = mock_parser

            yield mock_parser_module

    @pytest.fixture
    def mock_ahrs_filter(self):
        """Mock AHRS Madgwick filter and quaternion operations."""
        with (
            patch("pils.sensors.camera.Madgwick") as mock_madgwick,
            patch("pils.sensors.camera.acc2q") as mock_acc2q,
            patch("pils.sensors.camera.Quaternion") as mock_quat,
        ):
            # Mock Madgwick filter
            mock_filter = Mock()
            mock_filter.updateIMU.return_value = np.array([1.0, 0.0, 0.0, 0.0])
            mock_madgwick.return_value = mock_filter

            # Mock acc2q initial quaternion
            mock_acc2q.return_value = np.array([1.0, 0.0, 0.0, 0.0])

            # Mock Quaternion.to_angles() for Euler conversion
            mock_quat_instance = Mock()
            mock_quat_instance.to_angles.return_value = np.array([0.0, 0.1, 0.0])
            mock_quat.return_value = mock_quat_instance

            yield {
                "madgwick": mock_madgwick,
                "acc2q": mock_acc2q,
                "quaternion": mock_quat,
            }

    def test_sony_mode_finds_mp4_files(
        self, sony_video_and_log, mock_telemetry_parser, mock_ahrs_filter
    ):
        """Test that Sony mode finds .mp4 files (case-insensitive)."""
        camera_dir, _, _ = sony_video_and_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        _, model = camera.data
        assert model == "sony"

    def test_sony_mode_reads_log_file(
        self, sony_video_and_log, mock_telemetry_parser, mock_ahrs_filter
    ):
        """Test that Sony mode reads log file from parent directory."""
        camera_dir, _, log_file = sony_video_and_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        # Log file should be found and stored
        assert camera.logpath == log_file

    def test_sony_telemetry_parsing_gyro_accel(
        self, sony_video_and_log, mock_telemetry_parser, mock_ahrs_filter
    ):
        """Test extraction of gyro and accel data from Sony telemetry."""
        camera_dir, _, _ = sony_video_and_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        df, _ = camera.data

        # Should have gyro and accel columns
        assert "gyro_x" in df.columns
        assert "gyro_y" in df.columns
        assert "gyro_z" in df.columns
        assert "accel_x" in df.columns
        assert "accel_y" in df.columns
        assert "accel_z" in df.columns

        # Check values from mocked data
        assert len(df) == 3
        assert df["gyro_x"][0] == 0.1
        assert df["accel_z"][0] == 9.8

    def test_sony_telemetry_computes_quaternions(
        self, sony_video_and_log, mock_telemetry_parser, mock_ahrs_filter
    ):
        """Test that quaternions are computed."""
        camera_dir, _, _ = sony_video_and_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        df, _ = camera.data

        # Should have quaternion columns
        assert "qw" in df.columns
        assert "qx" in df.columns
        assert "qy" in df.columns
        assert "qz" in df.columns

        # Check quaternion values (from mocked filter)
        assert df["qw"][0] == 1.0

    def test_sony_telemetry_computes_euler_angles(
        self, sony_video_and_log, mock_telemetry_parser, mock_ahrs_filter
    ):
        """Test that Euler angles (roll, pitch, yaw) are computed."""
        camera_dir, _, _ = sony_video_and_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        df, _ = camera.data

        # Should have Euler angle columns
        assert "roll" in df.columns
        assert "pitch" in df.columns
        assert "yaw" in df.columns

        # Check values from mocked Quaternion.to_angles()
        assert df["pitch"][0] == 0.1

    def test_sony_timestamp_alignment(
        self, sony_video_and_log, mock_telemetry_parser, mock_ahrs_filter
    ):
        """Test that timestamps are aligned with log start time."""
        camera_dir, _, _ = sony_video_and_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        df, _ = camera.data

        # Should have timestamp column (aligned)
        assert "timestamp" in df.columns
        assert "timestamp_ms" in df.columns

        # timestamp should be timestamp_ms/1000 + log_start_time
        # Log time is 2024/11/20 14:30:45.123000
        # First timestamp_ms is 0, so timestamp should be log start time
        import datetime

        expected_start = datetime.datetime(2024, 11, 20, 14, 30, 45, 123000).timestamp()

        # Allow small floating point differences
        assert abs(df["timestamp"][0] - expected_start) < 0.001

    def test_sony_returns_model_string(
        self, sony_video_and_log, mock_telemetry_parser, mock_ahrs_filter
    ):
        """Test that Sony mode returns 'sony' model string."""
        camera_dir, _, _ = sony_video_and_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        _, model = camera.data
        assert model == "sony"


class TestCameraAlviumMode:
    """Test suite for Alvium industrial camera mode."""

    @pytest.fixture
    def alvium_log(self, tmp_path):
        """Create Alvium log file."""
        camera_dir = tmp_path / "alvium_camera"
        camera_dir.mkdir()

        log_file = camera_dir / "alvium.log"
        log_content = """[2024-11-20 14:30:15.123] INFO: Saving frame frame_0001.raw
[2024-11-20 14:30:15.223] INFO: Saving frame frame_0002.raw
[2024-11-20 14:30:15.323] INFO: Saving frame frame_0003.raw
[2024-11-20 14:30:15.423] INFO: Saving frame frame_0004.raw
"""
        log_file.write_text(log_content)

        return camera_dir, log_file

    def test_alvium_mode_when_no_video_files(self, alvium_log):
        """Test that Alvium mode activates when no .mp4 files found."""
        camera_dir, _ = alvium_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        _, model = camera.data
        assert model == "alvium"

    def test_alvium_log_parsing(self, alvium_log):
        """Test parsing of Alvium log file."""
        camera_dir, log_file = alvium_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        df, _ = camera.data

        # Should parse 4 frames
        assert len(df) == 4

        # Should have timestamp and frame_num columns
        assert "timestamp" in df.columns
        assert "frame_num" in df.columns

    def test_alvium_frame_extraction(self, alvium_log):
        """Test extraction of frame numbers."""
        camera_dir, _ = alvium_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        df, _ = camera.data

        # Check frame numbers
        assert df["frame_num"][0] == 1
        assert df["frame_num"][1] == 2
        assert df["frame_num"][2] == 3
        assert df["frame_num"][3] == 4

        # Check data types
        assert df["frame_num"].dtype == pl.Int64
        assert df["timestamp"].dtype == pl.Float64

    def test_alvium_returns_model_string(self, alvium_log):
        """Test that Alvium mode returns 'alvium' model string."""
        camera_dir, _ = alvium_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        _, model = camera.data
        assert model == "alvium"

    def test_alvium_timestamp_values(self, alvium_log):
        """Test that timestamps are correctly parsed."""
        camera_dir, _ = alvium_log

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        df, _ = camera.data

        # Check that timestamps are reasonable (Unix timestamps)
        # 2024-11-20 should be around 1.7e9 seconds
        assert df["timestamp"][0] > 1.7e9
        assert df["timestamp"][0] < 1.8e9

        # Check timestamp differences (should be ~0.1 seconds)
        ts_diff = df["timestamp"][1] - df["timestamp"][0]
        assert abs(ts_diff - 0.1) < 0.01  # Within 10ms


class TestCameraCaseInsensitiveExtensions:
    """Test case-insensitive file extension matching."""

    def test_uppercase_mp4_extension(
        self, tmp_path, mock_telemetry_parser=None, mock_ahrs_filter=None
    ):
        """Test that .MP4 files are also detected."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        # Create .MP4 file (uppercase)
        video_file = camera_dir / "video.MP4"
        video_file.write_bytes(b"fake_video")

        # Create log file
        log_file = tmp_path / "session.LOG"
        log_file.write_text(
            "2024/11/20 14:30:45.123000 [INFO] Camera Sony starts recording\n"
        )

        with patch("pils.sensors.camera.telemetry_parser") as mock_parser:
            mock_parser_inst = Mock()
            mock_parser_inst.normalized_imu.return_value = [
                {"timestamp_ms": 0, "gyro": [0, 0, 0], "accl": [0, 0, 9.8]}
            ]
            mock_parser.Parser.return_value = mock_parser_inst

            with (
                patch("pils.sensors.camera.Madgwick"),
                patch("pils.sensors.camera.acc2q", return_value=[1, 0, 0, 0]),
                patch("pils.sensors.camera.Quaternion") as mock_q,
            ):
                mock_q.return_value.to_angles.return_value = [0, 0, 0]

                camera = Camera(camera_dir, use_photogrammetry=False)
                camera.load_data()

                _, model = camera.data
                assert model == "sony"

    def test_lowercase_log_extension(self, tmp_path):
        """Test that .log files (lowercase) are detected."""
        camera_dir = tmp_path / "alvium"
        camera_dir.mkdir()

        log_file = camera_dir / "alvium.log"
        log_file.write_text(
            "[2024-11-20 14:30:15.123] INFO: Saving frame frame_0001.raw\n"
        )

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        _, model = camera.data
        assert model == "alvium"
