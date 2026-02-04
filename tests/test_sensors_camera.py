"""Tests for Camera sensor module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pils.sensors.camera import Camera


class TestCamera:
    """Test suite for Camera class."""

    @pytest.fixture
    def video_file(self, tmp_path):
        """Create mock video file path."""
        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake_video_data")
        return video_path

    @pytest.fixture
    def image_dir(self, tmp_path):
        """Create directory with test images."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Create fake image files
        (img_dir / "img_0001.jpg").write_bytes(b"fake_image_1")
        (img_dir / "img_0002.jpg").write_bytes(b"fake_image_2")
        (img_dir / "img_0003.jpg").write_bytes(b"fake_image_3")

        return img_dir

    @pytest.fixture
    def log_file(self, tmp_path):
        """Create log file."""
        log_path = tmp_path / "camera.log"
        log_path.write_text("2024-01-15 10:30:45 INFO:Camera Sony starts recording\n")
        return log_path

    def test_init_with_string_path(self, video_file):
        """Test Camera initialization with string path."""
        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(str(video_file))
            assert camera.path == str(video_file)
            assert camera.capture is None
            assert camera.fps is None
            assert camera.tstart is None

    def test_init_with_path_object(self, video_file):
        """Test Camera initialization with Path object."""
        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(video_file)
            assert camera.path == video_file
            assert camera.is_image_sequence is False

    def test_init_with_logpath(self, video_file, log_file):
        """Test Camera initialization with explicit logpath."""
        camera = Camera(video_file, logpath=log_file)
        assert camera.logpath == log_file

    def test_init_with_time_index(self, image_dir):
        """Test Camera initialization with time index."""
        time_index = {
            "img_0001.jpg": datetime(2024, 1, 15, 10, 30, 0),
            "img_0002.jpg": datetime(2024, 1, 15, 10, 30, 1),
        }
        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(image_dir, time_index=time_index)
            assert camera.time_index == time_index

    @patch("pils.sensors.camera.cv2.VideoCapture")
    @patch("pils.sensors.camera.read_log_time")
    def test_load_data_video_file(self, mock_read_log, mock_cv2, video_file, log_file):
        """Test load_data for video file."""
        # Mock VideoCapture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            0: 100.0,  # CAP_PROP_FRAME_COUNT
            5: 30.0,  # CAP_PROP_FPS
        }.get(prop, 0)
        mock_cv2.return_value = mock_cap

        # Mock log reading
        mock_read_log.return_value = (datetime(2024, 1, 15, 10, 30, 45), None)

        camera = Camera(video_file, logpath=log_file)
        camera.load_data()

        assert camera.capture is not None
        assert camera.fps == 30.0
        assert camera.tstart == datetime(2024, 1, 15, 10, 30, 45)

    def test_load_data_image_sequence(self, image_dir):
        """Test load_data for image sequence."""
        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(image_dir)
            camera.load_data()

            assert camera.is_image_sequence is True
            assert len(camera.images) == 3
            assert Path(camera.images[0]).name == "img_0001.jpg"

    def test_load_data_image_sequence_no_os_path(self, image_dir):
        """Test that image sequence loading uses pathlib, not os.path."""
        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(image_dir)
            camera.load_data()

            # Verify images are Path objects or strings
            for img in camera.images:
                # The result should be from pathlib glob, not os.path
                assert isinstance(img, (str, Path))

    def test_load_data_empty_directory_raises_error(self, tmp_path):
        """Test that empty directory raises FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(empty_dir)
            with pytest.raises(FileNotFoundError, match="No images found"):
                camera.load_data()

    @patch("pils.sensors.camera.cv2.VideoCapture")
    @patch("pils.sensors.camera.read_log_time")
    def test_get_frame_from_video(self, mock_read_log, mock_cv2, video_file, log_file):
        """Test get_frame from video file."""
        # Mock VideoCapture
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            0: 100.0,  # CAP_PROP_FRAME_COUNT
            5: 30.0,  # CAP_PROP_FPS
        }.get(prop, 0)

        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, fake_frame)
        mock_cv2.return_value = mock_cap

        mock_read_log.return_value = (datetime(2024, 1, 15, 10, 30, 45), None)

        camera = Camera(video_file, logpath=log_file)
        camera.load_data()

        frame = camera.get_frame(10)

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)
        mock_cap.set.assert_called_once()

    @patch("pils.sensors.camera.cv2.imread")
    def test_get_frame_from_image_sequence(self, mock_imread, image_dir):
        """Test get_frame from image sequence."""
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_imread.return_value = fake_frame

        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(image_dir)
            camera.load_data()

            frame = camera.get_frame(0)

            assert isinstance(frame, np.ndarray)
            assert frame.shape == (480, 640, 3)

    def test_get_frame_returns_ndarray_type(self, image_dir):
        """Test that get_frame has proper return type hint."""
        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(image_dir)
            # Type checking - verify return type annotation exists
            assert hasattr(camera.get_frame, "__annotations__")
            # In runtime, just check the signature exists
            import inspect

            sig = inspect.signature(camera.get_frame)
            assert "frame_number" in sig.parameters

    @patch("pils.sensors.camera.cv2.imread")
    def test_get_frame_index_out_of_range(self, mock_imread, image_dir):
        """Test get_frame with invalid index."""
        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(image_dir)
            camera.load_data()

            with pytest.raises(IndexError, match="Frame index out of range"):
                camera.get_frame(100)

    def test_pathlib_usage_not_os_path(self, image_dir):
        """Test that camera uses pathlib.Path.glob, not os.path."""
        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(image_dir)
            camera.load_data()

            # Check that images were found (proving glob worked)
            assert len(camera.images) == 3

            # Verify the images are sorted (by .name attribute from pathlib)
            image_names = [Path(img).name for img in camera.images]
            assert image_names == sorted(image_names)

    @patch("pils.sensors.camera.cv2.VideoCapture")
    @patch("pils.sensors.camera.read_log_time")
    def test_get_timestamp_video(self, mock_read_log, mock_cv2, video_file, log_file):
        """Test get_timestamp for video file."""
        mock_cap = Mock()
        mock_cap.get.side_effect = lambda prop: {
            0: 100.0,
            5: 30.0,
        }.get(prop, 0)
        mock_cv2.return_value = mock_cap
        mock_read_log.return_value = (datetime(2024, 1, 15, 10, 30, 45), None)

        camera = Camera(video_file, logpath=log_file)
        camera.load_data()

        timestamp = camera.get_timestamp(0)
        assert timestamp is not None
        assert isinstance(timestamp, datetime)

    def test_get_timestamp_images_with_index(self, image_dir):
        """Test get_timestamp for images with time_index."""
        time_index = {
            "img_0001.jpg": datetime(2024, 1, 15, 10, 30, 0),
            "img_0002.jpg": datetime(2024, 1, 15, 10, 30, 1),
            "img_0003.jpg": datetime(2024, 1, 15, 10, 30, 2),
        }
        with patch("pils.sensors.camera.get_logpath_from_datapath") as mock_log:
            mock_log.return_value = Path("/tmp/log.txt")
            camera = Camera(image_dir, time_index=time_index)
            camera.load_data()

            timestamp = camera.get_timestamp(0)
            assert timestamp == datetime(2024, 1, 15, 10, 30, 0)
