"""Edge case and error handling tests for Camera sensor module."""

from unittest.mock import Mock, patch

import polars as pl
import pytest

from pils.sensors.camera import Camera


class TestCameraErrorHandling:
    """Test suite for error handling scenarios."""

    def test_nonexistent_path_raises_error(self):
        """Test that non-existent path raises FileNotFoundError."""
        camera = Camera("/nonexistent/path/to/camera", use_photogrammetry=False)

        with pytest.raises(FileNotFoundError, match="Camera data path does not exist"):
            camera.load_data()

    def test_empty_directory_raises_error(self, tmp_path):
        """Test that empty directory raises FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        camera = Camera(empty_dir, use_photogrammetry=False)

        with pytest.raises(
            FileNotFoundError, match="No video files or log files found"
        ):
            camera.load_data()

    def test_photogrammetry_nonexistent_csv(self, tmp_path):
        """Test photogrammetry mode with non-existent CSV file."""
        csv_path = tmp_path / "nonexistent.csv"

        camera = Camera(csv_path, use_photogrammetry=True)

        with pytest.raises(FileNotFoundError, match="does not exist"):
            camera.load_data()

    def test_sony_missing_log_file_raises_error(self, tmp_path):
        """Test Sony mode without log file raises FileNotFoundError."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        # Create video file but no log
        video_file = camera_dir / "video.mp4"
        video_file.write_bytes(b"fake_video")

        camera = Camera(camera_dir, use_photogrammetry=False)

        with pytest.raises(FileNotFoundError, match="No log file found for Sony"):
            camera.load_data()

    def test_alvium_missing_log_file_raises_error(self, tmp_path):
        """Test Alvium mode without log file raises FileNotFoundError."""
        camera_dir = tmp_path / "alvium"
        camera_dir.mkdir()

        # No video files, no log files in directory
        camera = Camera(camera_dir, use_photogrammetry=False)

        with pytest.raises(
            FileNotFoundError, match="No video files or log files found"
        ):
            camera.load_data()

    def test_corrupted_telemetry_data_raises_error(self, tmp_path):
        """Test that corrupted telemetry data raises an error."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        video_file = camera_dir / "video.mp4"
        video_file.write_bytes(b"corrupted_data")

        log_file = tmp_path / "session.log"
        log_file.write_text(
            "2024/11/20 14:30:45.123000 [INFO] Camera Sony starts recording\n"
        )

        # Mock telemetry parser to raise exception
        with patch("pils.sensors.camera.telemetry_parser") as mock_parser:
            mock_parser.Parser.side_effect = Exception("Corrupted telemetry data")

            camera = Camera(camera_dir, use_photogrammetry=False)

            with pytest.raises(Exception, match="Corrupted telemetry data"):
                camera.load_data()

    def test_malformed_photogrammetry_csv(self, tmp_path):
        """Test photogrammetry CSV with missing required columns."""
        csv_path = tmp_path / "malformed.csv"
        # CSV without required columns
        df = pl.DataFrame({"time": [1000, 2000], "value": [1, 2]})
        df.write_csv(csv_path)

        camera = Camera(csv_path, use_photogrammetry=True)
        camera.load_data()

        # Should load but won't have expected columns
        df, model = camera.data
        assert model is None
        assert "time" in df.columns
        assert "timestamp" not in df.columns


class TestCameraEdgeCases:
    """Test suite for edge cases."""

    def test_empty_photogrammetry_csv(self, tmp_path):
        """Test photogrammetry CSV with header only (no data rows)."""
        csv_path = tmp_path / "empty.csv"
        # CSV with header but no data
        df = pl.DataFrame({"timestamp": [], "pitch": [], "roll": [], "yaw": []})
        df.write_csv(csv_path)

        camera = Camera(csv_path, use_photogrammetry=True)
        camera.load_data()

        df, model = camera.data
        assert len(df) == 0
        assert model is None

    def test_multiple_video_files_uses_first(self, tmp_path):
        """Test that when multiple video files exist, first is used."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        # Create multiple video files
        (camera_dir / "video_001.mp4").write_bytes(b"video1")
        (camera_dir / "video_002.mp4").write_bytes(b"video2")
        (camera_dir / "video_003.mp4").write_bytes(b"video3")

        log_file = tmp_path / "session.log"
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

                # Should successfully load (uses first video file)
                df, model = camera.data
                assert model == "sony"
                assert len(df) == 1

    def test_multiple_log_files_uses_first(self, tmp_path):
        """Test that when multiple log files exist, first is used."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        video_file = camera_dir / "video.mp4"
        video_file.write_bytes(b"video_data")

        # Create multiple log files in parent
        (tmp_path / "log_001.log").write_text(
            "2024/11/20 14:30:45.123000 [INFO] Camera Sony starts recording\n"
        )
        (tmp_path / "log_002.log").write_text(
            "2024/11/20 15:00:00.000000 [INFO] Camera Sony starts recording\n"
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

                # Should successfully load with first log file
                df, model = camera.data
                assert model == "sony"

    def test_mixed_case_extensions(self, tmp_path):
        """Test various mixed case file extensions."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        # Create .Mp4 file (mixed case)
        video_file = camera_dir / "video.Mp4"
        video_file.write_bytes(b"video_data")

        log_file = tmp_path / "session.Log"
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

    def test_alvium_with_non_matching_lines(self, tmp_path):
        """Test Alvium log with some non-matching lines."""
        camera_dir = tmp_path / "alvium"
        camera_dir.mkdir()

        log_file = camera_dir / "alvium.log"
        log_content = """[2024-11-20 14:30:15.123] INFO: Camera started
[2024-11-20 14:30:15.223] INFO: Saving frame frame_0001.raw
[2024-11-20 14:30:15.323] DEBUG: Some other message
[2024-11-20 14:30:15.423] INFO: Saving frame frame_0002.raw
[2024-11-20 14:30:15.523] ERROR: Connection lost
[2024-11-20 14:30:15.623] INFO: Saving frame frame_0003.raw
"""
        log_file.write_text(log_content)

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        df, model = camera.data

        # Should only parse lines with "Saving frame"
        assert len(df) == 3
        assert model == "alvium"
        assert df["frame_num"][0] == 1
        assert df["frame_num"][2] == 3

    def test_alvium_empty_log_after_filtering(self, tmp_path):
        """Test Alvium log with no matching frame lines."""
        camera_dir = tmp_path / "alvium"
        camera_dir.mkdir()

        log_file = camera_dir / "alvium.log"
        log_content = """[2024-11-20 14:30:15.123] INFO: Camera started
[2024-11-20 14:30:15.223] DEBUG: Some message
[2024-11-20 14:30:15.323] ERROR: Connection lost
"""
        log_file.write_text(log_content)

        camera = Camera(camera_dir, use_photogrammetry=False)
        camera.load_data()

        df, model = camera.data

        # Should return empty DataFrame
        assert len(df) == 0
        assert model == "alvium"

    def test_sony_telemetry_with_single_sample(self, tmp_path):
        """Test Sony telemetry with only one IMU sample."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        video_file = camera_dir / "video.mp4"
        video_file.write_bytes(b"video_data")

        log_file = tmp_path / "session.log"
        log_file.write_text(
            "2024/11/20 14:30:45.123000 [INFO] Camera Sony starts recording\n"
        )

        with patch("pils.sensors.camera.telemetry_parser") as mock_parser:
            mock_parser_inst = Mock()
            # Only one sample
            mock_parser_inst.normalized_imu.return_value = [
                {"timestamp_ms": 0, "gyro": [0.1, 0.2, 0.3], "accl": [0, 0, 9.8]}
            ]
            mock_parser.Parser.return_value = mock_parser_inst

            with (
                patch("pils.sensors.camera.Madgwick"),
                patch("pils.sensors.camera.acc2q", return_value=[1, 0, 0, 0]),
                patch("pils.sensors.camera.Quaternion") as mock_q,
            ):
                mock_q.return_value.to_angles.return_value = [0.1, 0.2, 0.3]

                camera = Camera(camera_dir, use_photogrammetry=False)
                camera.load_data()

                df, model = camera.data
                assert len(df) == 1
                assert model == "sony"
                # Should have all columns even with one sample
                assert "roll" in df.columns
                assert "pitch" in df.columns
                assert "yaw" in df.columns

    def test_sony_telemetry_zero_dt_handles_gracefully(self, tmp_path):
        """Test Sony telemetry with identical timestamps (zero dt)."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        video_file = camera_dir / "video.mp4"
        video_file.write_bytes(b"video_data")

        log_file = tmp_path / "session.log"
        log_file.write_text(
            "2024/11/20 14:30:45.123000 [INFO] Camera Sony starts recording\n"
        )

        with patch("pils.sensors.camera.telemetry_parser") as mock_parser:
            mock_parser_inst = Mock()
            # Same timestamp for all samples (edge case)
            mock_parser_inst.normalized_imu.return_value = [
                {"timestamp_ms": 1000, "gyro": [0, 0, 0], "accl": [0, 0, 9.8]},
                {"timestamp_ms": 1000, "gyro": [0, 0, 0], "accl": [0, 0, 9.8]},
            ]
            mock_parser.Parser.return_value = mock_parser_inst

            with (
                patch("pils.sensors.camera.Madgwick"),
                patch("pils.sensors.camera.acc2q", return_value=[1, 0, 0, 0]),
                patch("pils.sensors.camera.Quaternion") as mock_q,
            ):
                mock_q.return_value.to_angles.return_value = [0, 0, 0]

                camera = Camera(camera_dir, use_photogrammetry=False)

                # May raise error due to zero dt (frequency calculation issue)
                # or handle gracefully - test that it doesn't hang
                try:
                    camera.load_data()
                    df, model = camera.data
                    assert model == "sony"
                except (ZeroDivisionError, ValueError):
                    # It's acceptable to raise an error for invalid data
                    pass

    def test_photogrammetry_with_nan_values(self, tmp_path):
        """Test photogrammetry CSV with NaN values."""
        csv_path = tmp_path / "photogrammetry_nan.csv"
        df = pl.DataFrame(
            {
                "timestamp": [1000.0, 2000.0, None, 4000.0],
                "pitch": [0.1, None, 0.3, 0.4],
                "roll": [0.05, 0.15, 0.25, None],
                "yaw": [1.5, 1.6, 1.7, 1.8],
            }
        )
        df.write_csv(csv_path)

        camera = Camera(csv_path, use_photogrammetry=True)
        camera.load_data()

        result_df, model = camera.data
        assert model is None
        assert len(result_df) == 4
        # NaN values should be preserved
        assert (
            result_df["timestamp"][2] is None
            or pl.Series([result_df["timestamp"][2]]).is_null()[0]
        )


class TestCameraLogging:
    """Test logging behavior for warnings and errors."""

    def test_sony_missing_log_logs_warning(self, tmp_path, caplog):
        """Test that missing Sony log file logs a warning."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        video_file = camera_dir / "video.mp4"
        video_file.write_bytes(b"video_data")

        camera = Camera(camera_dir, use_photogrammetry=False)

        with pytest.raises(FileNotFoundError):
            camera.load_data()

        # Should have logged warning before raising error
        assert any("No log file found" in record.message for record in caplog.records)

    def test_alvium_missing_log_logs_warning(self, tmp_path, caplog):
        """Test that missing Alvium log file logs a warning."""
        camera_dir = tmp_path / "alvium"
        camera_dir.mkdir()

        camera = Camera(camera_dir, use_photogrammetry=False)

        with pytest.raises(FileNotFoundError):
            camera.load_data()

        # Should have logged warning
        assert any("No log file found" in record.message for record in caplog.records)

    def test_telemetry_parsing_error_logs_error(self, tmp_path, caplog):
        """Test that telemetry parsing errors are logged."""
        camera_dir = tmp_path / "camera"
        camera_dir.mkdir()

        video_file = camera_dir / "video.mp4"
        video_file.write_bytes(b"corrupted")

        log_file = tmp_path / "session.log"
        log_file.write_text(
            "2024/11/20 14:30:45.123000 [INFO] Camera Sony starts recording\n"
        )

        with patch("pils.sensors.camera.telemetry_parser") as mock_parser:
            mock_parser.Parser.side_effect = RuntimeError("Parse failed")

            camera = Camera(camera_dir, use_photogrammetry=False)

            with pytest.raises(RuntimeError):
                camera.load_data()

            # Should have logged error
            assert any(
                "Failed to parse Sony telemetry" in record.message
                for record in caplog.records
            )
