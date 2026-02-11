"""Integration tests for Camera with Flight and Synchronizer classes."""

from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import pytest

from pils.flight import Flight


class TestCameraFlightIntegration:
    """Test suite for Camera integration with Flight class."""

    @pytest.fixture
    def mock_flight_info_photogrammetry(self, tmp_path):
        """Create flight_info with photogrammetry folder."""
        proc_data = tmp_path / "proc_data"
        proc_data.mkdir()
        photogrammetry_dir = proc_data / "photogrammetry"
        photogrammetry_dir.mkdir()

        # Create photogrammetry CSV
        csv_path = photogrammetry_dir / "photogrammetry.csv"
        df = pl.DataFrame(
            {
                "timestamp": [1000.0, 2000.0, 3000.0],
                "pitch": [0.1, 0.2, 0.3],
                "roll": [0.05, 0.15, 0.25],
                "yaw": [1.5, 1.6, 1.7],
            }
        )
        df.write_csv(csv_path)

        return {
            "proc_data_folder_path": str(proc_data),
            "drone_data_folder_path": str(tmp_path / "drone"),
        }

    @pytest.fixture
    def mock_flight_info_sony(self, tmp_path):
        """Create flight_info with Sony camera folder."""
        aux_data = tmp_path / "aux_data"
        aux_data.mkdir()
        camera_dir = aux_data / "camera"
        camera_dir.mkdir()

        # Create video file
        video_file = camera_dir / "video.mp4"
        video_file.write_bytes(b"fake_video")

        # Create log file in parent (aux_data)
        log_file = aux_data / "session.log"
        log_file.write_text(
            "2024/11/20 14:30:45.123000 [INFO] Camera Sony starts recording\n"
        )

        return {
            "aux_data_folder_path": str(aux_data),
            "drone_data_folder_path": str(tmp_path / "drone"),
        }

    @pytest.fixture
    def mock_flight_info_alvium(self, tmp_path):
        """Create flight_info with Alvium camera folder."""
        aux_data = tmp_path / "aux_data"
        aux_data.mkdir()
        camera_dir = aux_data / "camera"
        camera_dir.mkdir()

        # Create Alvium log file
        log_file = camera_dir / "alvium.log"
        log_content = """[2024-11-20 14:30:15.123] INFO: Saving frame frame_0001.raw
[2024-11-20 14:30:15.223] INFO: Saving frame frame_0002.raw
[2024-11-20 14:30:15.323] INFO: Saving frame frame_0003.raw
"""
        log_file.write_text(log_content)

        return {
            "aux_data_folder_path": str(aux_data),
            "drone_data_folder_path": str(tmp_path / "drone"),
        }

    def test_flight_add_camera_photogrammetry(self, mock_flight_info_photogrammetry):
        """Test Flight.add_camera_data() with photogrammetry mode."""
        flight = Flight(mock_flight_info_photogrammetry)
        flight.add_camera_data(use_photogrammetry=True)

        # Verify camera data was loaded into flight
        assert flight.raw_data.payload_data.camera is not None
        camera_df = flight.raw_data.payload_data.camera

        # Should be a DataFrame
        assert isinstance(camera_df, pl.DataFrame)
        assert len(camera_df) == 3

        # Check columns
        assert "timestamp" in camera_df.columns
        assert "pitch" in camera_df.columns

    def test_flight_add_camera_sony_mode(self, mock_flight_info_sony):
        """Test Flight.add_camera_data() with Sony camera mode."""
        with patch("pils.sensors.camera.telemetry_parser") as mock_parser:
            mock_parser_inst = Mock()
            mock_parser_inst.normalized_imu.return_value = [
                {"timestamp_ms": 0, "gyro": [0.1, 0.2, 0.3], "accl": [0, 0, 9.8]},
                {
                    "timestamp_ms": 100,
                    "gyro": [0.15, 0.25, 0.35],
                    "accl": [0.1, 0.1, 9.7],
                },
            ]
            mock_parser.Parser.return_value = mock_parser_inst

            with (
                patch("pils.sensors.camera.Madgwick") as mock_madgwick_class,
                patch("pils.sensors.camera.acc2q", return_value=[1, 0, 0, 0]),
                patch("pils.sensors.camera.Quaternion") as mock_q,
            ):
                # Setup Madgwick mock
                mock_madgwick = Mock()
                mock_madgwick.updateIMU.return_value = np.array([1, 0, 0, 0])
                mock_madgwick_class.return_value = mock_madgwick

                # Setup Quaternion mock
                mock_q.return_value.to_angles.return_value = [0.1, 0.2, 0.3]

                flight = Flight(mock_flight_info_sony)
                flight.add_camera_data(use_photogrammetry=False)

                # Verify camera data was loaded
                assert flight.raw_data.payload_data.camera is not None
                camera_df = flight.raw_data.payload_data.camera

                # Should have Sony telemetry columns
                assert "gyro_x" in camera_df.columns
                assert "accel_x" in camera_df.columns
                assert "roll" in camera_df.columns
                assert "pitch" in camera_df.columns
                assert "yaw" in camera_df.columns
                assert "timestamp" in camera_df.columns

    def test_flight_add_camera_alvium_mode(self, mock_flight_info_alvium):
        """Test Flight.add_camera_data() with Alvium camera mode."""
        flight = Flight(mock_flight_info_alvium)
        flight.add_camera_data(use_photogrammetry=False)

        # Verify camera data was loaded
        assert flight.raw_data.payload_data.camera is not None
        camera_df = flight.raw_data.payload_data.camera

        # Should have Alvium columns
        assert "timestamp" in camera_df.columns
        assert "frame_num" in camera_df.columns
        assert len(camera_df) == 3

    def test_flight_camera_data_stored_in_raw_data(
        self, mock_flight_info_photogrammetry
    ):
        """Test that camera data is properly stored in flight.raw_data hierarchy."""
        flight = Flight(mock_flight_info_photogrammetry)
        flight.add_camera_data(use_photogrammetry=True)

        # Check data flow: Camera → Flight.raw_data.payload_data.camera
        assert hasattr(flight.raw_data, "payload_data")
        assert hasattr(flight.raw_data.payload_data, "camera")
        assert flight.raw_data.payload_data.camera is not None

    def test_flight_camera_model_attribute(self, mock_flight_info_sony):
        """Test that Flight stores camera model type."""
        with patch("pils.sensors.camera.telemetry_parser") as mock_parser:
            mock_parser_inst = Mock()
            mock_parser_inst.normalized_imu.return_value = [
                {"timestamp_ms": 0, "gyro": [0, 0, 0], "accl": [0, 0, 9.8]}
            ]
            mock_parser.Parser.return_value = mock_parser_inst

            with (
                patch("pils.sensors.camera.Madgwick") as mock_madgwick_class,
                patch("pils.sensors.camera.acc2q", return_value=[1, 0, 0, 0]),
                patch("pils.sensors.camera.Quaternion") as mock_q,
            ):
                # Setup Madgwick mock
                mock_madgwick = Mock()
                mock_madgwick.updateIMU.return_value = np.array([1, 0, 0, 0])
                mock_madgwick_class.return_value = mock_madgwick

                # Setup Quaternion mock
                mock_q.return_value.to_angles.return_value = [0, 0, 0]

                flight = Flight(mock_flight_info_sony)
                flight.add_camera_data(use_photogrammetry=False)

                # Check private attribute for camera model
                assert hasattr(flight, "_Flight__camera_model")
                assert flight._Flight__camera_model == "sony"


class TestCameraSynchronizerIntegration:
    """Test suite for Camera integration with Synchronizer class."""

    @pytest.fixture
    def mock_gps_data(self):
        """Create mock GPS data for synchronizer."""
        return pl.DataFrame(
            {
                "timestamp": [1000.0, 1001.0, 1002.0, 1003.0, 1004.0],
                "lat": [40.7128, 40.7129, 40.7130, 40.7131, 40.7132],
                "lon": [-74.0060, -74.0061, -74.0062, -74.0063, -74.0064],
                "altitude": [100.0, 101.0, 102.0, 103.0, 104.0],
            }
        )

    def test_synchronizer_add_camera_photogrammetry(self, mock_gps_data):
        """Test Synchronizer.add_camera() with photogrammetry data."""
        from pils.synchronizer import Synchronizer

        camera_data = pl.DataFrame(
            {
                "timestamp": [1000.5, 1001.5, 1002.5, 1003.5],
                "pitch": [0.1, 0.2, 0.3, 0.4],
                "roll": [0.05, 0.15, 0.25, 0.35],
            }
        )

        sync = Synchronizer()
        sync.add_gps_reference(
            mock_gps_data,
            timestamp_col="timestamp",
            lat_col="lat",
            lon_col="lon",
            alt_col="altitude",
        )
        sync.add_camera(camera_data, use_photogrammetry=True)

        # Verify camera data was added
        assert sync.camera is not None
        assert len(sync.camera) == 4

    def test_synchronizer_add_camera_sony(self, mock_gps_data):
        """Test Synchronizer.add_camera() with Sony camera data."""
        from pils.synchronizer import Synchronizer

        camera_data = pl.DataFrame(
            {
                "timestamp": [1000.5, 1001.5, 1002.5],
                "pitch": [0.1, 0.2, 0.3],
                "roll": [0.05, 0.15, 0.25],
                "yaw": [1.5, 1.6, 1.7],
                "gyro_x": [0.1, 0.2, 0.3],
                "gyro_y": [0.1, 0.2, 0.3],
                "gyro_z": [0.1, 0.2, 0.3],
            }
        )

        sync = Synchronizer()
        sync.add_gps_reference(
            mock_gps_data,
            timestamp_col="timestamp",
            lat_col="lat",
            lon_col="lon",
            alt_col="altitude",
        )
        sync.add_camera(camera_data, use_photogrammetry=False, camera_model="sony")

        # Verify camera data was added
        assert sync.camera is not None
        assert len(sync.camera) == 3

    def test_camera_data_flow_end_to_end(self, tmp_path):
        """Test complete data flow: Camera → Flight → raw_data.payload_data.camera."""
        # Create photogrammetry data
        proc_data = tmp_path / "proc_data"
        proc_data.mkdir()
        photogrammetry_dir = proc_data / "photogrammetry"
        photogrammetry_dir.mkdir()

        csv_path = photogrammetry_dir / "photogrammetry.csv"
        df = pl.DataFrame(
            {
                "timestamp": [1000.0, 2000.0, 3000.0],
                "pitch": [0.1, 0.2, 0.3],
                "roll": [0.05, 0.15, 0.25],
                "yaw": [1.5, 1.6, 1.7],
            }
        )
        df.write_csv(csv_path)

        flight_info = {
            "proc_data_folder_path": str(proc_data),
            "drone_data_folder_path": str(tmp_path / "drone"),
        }

        # Complete flow
        flight = Flight(flight_info)
        flight.add_camera_data(use_photogrammetry=True)

        # Verify data at each step
        # 1. Camera loads data
        # 2. Flight.add_camera_data() calls Camera.load_data()
        # 3. Data stored in flight.raw_data.payload_data.camera
        camera_df = flight.raw_data.payload_data.camera
        assert camera_df is not None
        assert isinstance(camera_df, pl.DataFrame)
        assert len(camera_df) == 3
        assert "pitch" in camera_df.columns

        # Verify original data integrity
        assert camera_df["pitch"][0] == 0.1
        assert camera_df["timestamp"][0] == 1000.0


class TestCameraFlightSyncIntegration:
    """Test suite for Camera with Flight.sync() integration."""

    @pytest.fixture
    def flight_with_camera_and_gps(self, tmp_path):
        """Create Flight with camera and GPS data."""
        # Setup aux_data folder
        aux_data = tmp_path / "aux_data"
        aux_data.mkdir()

        # Create camera folder with Alvium log
        camera_dir = aux_data / "camera"
        camera_dir.mkdir()
        log_file = camera_dir / "alvium.log"
        log_content = """[2024-11-20 14:30:15.123] INFO: Saving frame frame_0001.raw
[2024-11-20 14:30:15.223] INFO: Saving frame frame_0002.raw
[2024-11-20 14:30:15.323] INFO: Saving frame frame_0003.raw
"""
        log_file.write_text(log_content)

        # Create sensors folder with GPS data
        sensors_dir = aux_data / "sensors"
        sensors_dir.mkdir()
        gps_dir = sensors_dir / "gps"
        gps_dir.mkdir()

        # Create GPS CSV
        gps_csv = gps_dir / "gps.csv"
        gps_df = pl.DataFrame(
            {
                "timestamp": [
                    1732112415.123,
                    1732112415.223,
                    1732112415.323,
                    1732112415.423,
                ],
                "latitude": [40.7128, 40.7129, 40.7130, 40.7131],
                "longitude": [-74.0060, -74.0061, -74.0062, -74.0063],
                "altitude": [100.0, 101.0, 102.0, 103.0],
            }
        )
        gps_df.write_csv(gps_csv)

        flight_info = {
            "aux_data_folder_path": str(aux_data),
            "drone_data_folder_path": str(tmp_path / "drone"),
        }

        return Flight(flight_info)

    def test_flight_with_camera_data_structure(self, flight_with_camera_and_gps):
        """Test that camera data is correctly accessible from Flight."""
        flight = flight_with_camera_and_gps

        # Load camera only (don't load GPS for this test)
        flight.add_camera_data(use_photogrammetry=False)

        # Verify camera data is loaded
        assert flight.raw_data.payload_data.camera is not None

        # Verify camera data is a DataFrame
        camera_df = flight.raw_data.payload_data.camera
        assert isinstance(camera_df, pl.DataFrame)
        assert "timestamp" in camera_df.columns
        assert "frame_num" in camera_df.columns

    def test_camera_dataframe_structure(self, tmp_path):
        """Test that camera DataFrame has correct structure for synchronization."""
        # Create simple photogrammetry data
        proc_data = tmp_path / "proc_data"
        proc_data.mkdir()
        photo_dir = proc_data / "photogrammetry"
        photo_dir.mkdir()

        csv_path = photo_dir / "photo.csv"
        df = pl.DataFrame(
            {
                "timestamp": [1000.0, 2000.0, 3000.0],
                "pitch": [0.1, 0.2, 0.3],
            }
        )
        df.write_csv(csv_path)

        flight_info = {
            "proc_data_folder_path": str(proc_data),
            "drone_data_folder_path": str(tmp_path / "drone"),
        }

        flight = Flight(flight_info)
        flight.add_camera_data(use_photogrammetry=True)

        camera_df = flight.raw_data.payload_data.camera

        # Verify structure needed for synchronization
        assert "timestamp" in camera_df.columns
        assert camera_df["timestamp"].dtype in [pl.Float64, pl.Int64]
        assert len(camera_df) > 0
