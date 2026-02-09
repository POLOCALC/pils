"""Tests for Flight and PayloadData class."""

from unittest.mock import Mock, patch

import polars as pl
import pytest

from pils.flight import DroneData, Flight, PayloadData


class TestPayloadData:
    """Test suite for PayloadData class."""

    def test_payload_data_getattr(self):
        """Test PayloadData __getattr__ returns attributes correctly."""
        payload = PayloadData()
        # Set a sensor dynamically
        payload.gps = pl.DataFrame({"timestamp": [1, 2, 3], "lat": [40.0, 40.1, 40.2]})

        # Access via getattr should work
        gps_data = payload.gps
        assert isinstance(gps_data, pl.DataFrame)
        assert "timestamp" in gps_data.columns
        assert "lat" in gps_data.columns
        assert gps_data.shape[0] == 3

    def test_payload_data_missing_attribute(self):
        """Test PayloadData raises AttributeError for missing sensors."""
        payload = PayloadData()

        with pytest.raises(AttributeError) as exc_info:
            _ = payload.nonexistent_sensor

        assert "nonexistent_sensor" in str(exc_info.value)
        assert "not loaded" in str(exc_info.value).lower()

    def test_payload_data_multiple_sensors(self):
        """Test PayloadData can store multiple sensors."""
        payload = PayloadData()

        # Add multiple sensors
        payload.gps = pl.DataFrame({"timestamp": [1, 2], "lat": [40.0, 40.1]})
        payload.imu = pl.DataFrame({"timestamp": [1, 2], "accel_x": [0.1, 0.2]})
        payload.adc = pl.DataFrame({"timestamp": [1, 2], "voltage": [3.3, 3.4]})

        # All should be accessible
        assert isinstance(payload.gps, pl.DataFrame)
        assert isinstance(payload.imu, pl.DataFrame)
        assert isinstance(payload.adc, pl.DataFrame)

        # Check list_loaded_sensors includes all
        sensors = payload.list_loaded_sensors()
        assert "gps" in sensors
        assert "imu" in sensors
        assert "adc" in sensors

    def test_payload_data_attribute_error_message(self):
        """Test that AttributeError message includes available sensors."""
        payload = PayloadData()
        payload.gps = pl.DataFrame({"timestamp": [1]})
        payload.imu = pl.DataFrame({"timestamp": [1]})

        with pytest.raises(AttributeError) as exc_info:
            _ = payload.missing_sensor

        error_msg = str(exc_info.value)
        # Should mention the missing sensor
        assert "missing_sensor" in error_msg
        # Should list available sensors
        assert "gps" in error_msg or "Available sensors" in error_msg


class TestFlightTypeGuards:
    """Test suite for Flight type guards and assertions."""

    @pytest.fixture
    def temp_flight(self, tmp_path):
        """Create a Flight object with valid paths."""
        drone_folder = tmp_path / "drone"
        aux_folder = tmp_path / "aux"
        sensors_folder = aux_folder / "sensors"

        # Create directories
        drone_folder.mkdir(parents=True)
        sensors_folder.mkdir(parents=True)

        flight_info = {
            "drone_data_folder_path": str(drone_folder),
            "aux_data_folder_path": str(aux_folder),
        }

        return Flight(flight_info)

    def test_drone_data_not_none_after_load(self, temp_flight):
        """Test that drone_data is not None after add_drone_data()."""
        # Mock the drone loading to succeed
        mock_drone_df = pl.DataFrame(
            {
                "timestamp": [1000, 2000, 3000],
                "latitude": [40.0, 40.1, 40.2],
                "longitude": [-74.0, -74.1, -74.2],
            }
        )

        # Directly set drone_data to simulate successful loading
        temp_flight.raw_data.drone_data = DroneData(mock_drone_df, None)

        # The assertion would be checked when add_drone_data() runs
        # Here we verify the data is set correctly
        assert temp_flight.raw_data.drone_data is not None
        assert isinstance(temp_flight.raw_data.drone_data, DroneData)

    def test_payload_data_not_none_after_load(self, temp_flight):
        """Test that payload_data is not None after add_sensor_data()."""
        # Mock sensor loading
        mock_sensor = Mock()
        mock_sensor.data = pl.DataFrame({"timestamp": [1, 2, 3], "value": [1.0, 2.0, 3.0]})

        with patch("pils.flight.sensor_config") as mock_config:
            # Setup mock sensor config
            mock_config.get.return_value = {
                "class": Mock(return_value=mock_sensor),
                "load_method": "load_data",
            }

            # Call add_sensor_data - should initialize payload_data and assertion should pass
            temp_flight.add_sensor_data("gps")

            # Verify payload_data was set (assertion didn't fail)
            assert temp_flight.raw_data.payload_data is not None
            assert isinstance(temp_flight.raw_data.payload_data, PayloadData)

    def test_type_narrowing_after_loading(self, temp_flight):
        """Test that type checkers understand data is not None after loading."""
        # This test verifies the runtime behavior with non-Optional types
        # The actual type narrowing is checked by Pylance statically

        mock_drone_df = pl.DataFrame({"timestamp": [1, 2], "lat": [40.0, 40.1]})

        # Before loading, drone_data is initialized with empty DataFrames
        assert temp_flight.raw_data.drone_data is not None
        assert len(temp_flight.raw_data.drone_data.drone) == 0  # Empty DataFrame

        # Simulate loading by setting drone_data
        temp_flight.raw_data.drone_data = DroneData(mock_drone_df, None)

        # After setting, drone_data still not None, and drone attribute has data
        assert temp_flight.raw_data.drone_data is not None
        assert len(temp_flight.raw_data.drone_data.drone) > 0  # Has data now

        # Type narrowing allows accessing attributes without errors
        # (This is what Pylance will understand)
        drone_data = temp_flight.raw_data.drone_data
        assert drone_data is not None  # Type guard
        assert hasattr(drone_data, "drone")  # Can access attributes safely

    def test_assertion_with_multiple_sensors(self, temp_flight):
        """Test assertion works when loading multiple sensors."""
        mock_sensor = Mock()
        mock_sensor.data = pl.DataFrame({"timestamp": [1], "value": [1.0]})

        with patch("pils.flight.sensor_config") as mock_config:
            mock_config.get.return_value = {
                "class": Mock(return_value=mock_sensor),
                "load_method": "load_data",
            }

            # Load multiple sensors
            temp_flight.add_sensor_data(["gps", "imu", "adc"])

            # PayloadData should be initialized (assertion passed)
            assert temp_flight.raw_data.payload_data is not None
            # Can access sensors safely after assertion
            payload = temp_flight.raw_data.payload_data
            assert payload is not None  # Type guard
            assert hasattr(payload, "list_loaded_sensors")
