"""Tests for the common_time configuration in Synchronizer and Flight.sync()."""

import numpy as np
import polars as pl
import pytest

from pils.flight import Flight, PayloadData
from pils.synchronizer import Synchronizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_gps():
    """Create a GPS payload reference spanning 0-10 seconds at 1 Hz."""
    t = np.linspace(0.0, 10.0, 11)
    return pl.DataFrame(
        {
            "timestamp": t,
            "posllh_lat": 45.0 + 0.0001 * t,
            "posllh_lon": 10.0 + 0.0001 * t,
            "posllh_height": 100.0 + t,
        }
    )


@pytest.fixture
def drone_gps_with_offset(reference_gps):
    """
    Create drone GPS data with a known 2-second time offset and enough overlap.

    The drone data spans 2-12 seconds (shifted by +2 s) so the window
    2-10 s overlaps at least 10 s with the reference.
    """
    t = np.linspace(2.0, 120.0, 1180)  # long, 100 Hz data, 2 s ahead
    return pl.DataFrame(
        {
            "correct_timestamp": t,
            "latitude": 45.0 + 0.0001 * (t - 2.0),
            "longitude": 10.0 + 0.0001 * (t - 2.0),
            "altitude": 100.0 + (t - 2.0),
        }
    )


@pytest.fixture
def payload_sensor():
    """Create a simple ADC-like sensor DataFrame at 10 Hz."""
    t = np.linspace(0.0, 10.0, 101)
    return pl.DataFrame(
        {
            "timestamp": t,
            "voltage": 3.3 + 0.01 * np.sin(t),
        }
    )


@pytest.fixture
def basic_synchronizer(reference_gps):
    """Synchronizer with only the GPS reference set (no offsets computed yet)."""
    sync = Synchronizer()
    sync.add_gps_reference(reference_gps)
    return sync


# ---------------------------------------------------------------------------
# Synchronizer.synchronize() – common_time=True (default)
# ---------------------------------------------------------------------------


class TestSynchronizerCommonTimeTrue:
    """common_time=True: all outputs share an evenly-spaced time axis."""

    def test_payload_timestamps_are_uniform(self, basic_synchronizer, payload_sensor):
        """Payload sensor timestamps must be uniformly spaced when common_time=True."""
        basic_synchronizer.add_payload_sensor("adc", payload_sensor)
        result = basic_synchronizer.synchronize(
            target_rate={"drone": 10.0, "payload": 10.0},
            common_time=True,
        )
        adc_ts = result["adc_timestamp"].to_numpy()
        diffs = np.diff(adc_ts)
        assert np.allclose(diffs, diffs[0], rtol=1e-6), (
            "Timestamps should be uniformly spaced with common_time=True"
        )

    def test_payload_timestamps_span_reference_window(
        self, basic_synchronizer, payload_sensor
    ):
        """Payload timestamps must cover the reference GPS time window."""
        basic_synchronizer.add_payload_sensor("adc", payload_sensor)
        result = basic_synchronizer.synchronize(
            target_rate={"drone": 10.0, "payload": 10.0},
            common_time=True,
        )
        ref_ts = basic_synchronizer.gps_payload["timestamp"].to_numpy()
        adc_ts = result["adc_timestamp"].to_numpy()
        assert adc_ts[0] == pytest.approx(ref_ts[0], abs=1e-6)
        assert adc_ts[-1] == pytest.approx(ref_ts[-1], abs=0.2)

    def test_payload_sample_count_matches_target_rate(
        self, basic_synchronizer, payload_sensor
    ):
        """Number of output samples should match target_rate * duration (approx)."""
        target_rate = 5.0
        basic_synchronizer.add_payload_sensor("adc", payload_sensor)
        result = basic_synchronizer.synchronize(
            target_rate={"drone": 10.0, "payload": target_rate},
            common_time=True,
        )
        ref_ts = basic_synchronizer.gps_payload["timestamp"].to_numpy()
        duration = ref_ts[-1] - ref_ts[0]
        expected_n = int(duration * target_rate) + 1
        adc_ts = result["adc_timestamp"].to_numpy()
        assert len(adc_ts) == expected_n

    def test_reference_gps_always_included(self, basic_synchronizer, payload_sensor):
        """reference_gps must be present in the output regardless of common_time."""
        basic_synchronizer.add_payload_sensor("adc", payload_sensor)
        result = basic_synchronizer.synchronize(
            target_rate={"drone": 10.0, "payload": 10.0},
            common_time=True,
        )
        assert "reference_gps" in result
        assert isinstance(result["reference_gps"], pl.DataFrame)


# ---------------------------------------------------------------------------
# Synchronizer.synchronize() – common_time=False
# ---------------------------------------------------------------------------


class TestSynchronizerCommonTimeFalse:
    """common_time=False: only timestamps are shifted, values are kept as-is."""

    def test_payload_sensor_values_unchanged(self, basic_synchronizer, payload_sensor):
        """Sensor values must not be interpolated when common_time=False."""
        basic_synchronizer.add_payload_sensor("adc", payload_sensor)
        result = basic_synchronizer.synchronize(
            target_rate={"drone": 10.0, "payload": 10.0},
            common_time=False,
        )
        # Values should be identical to the original sensor data
        original_values = payload_sensor["voltage"].to_numpy().flatten()
        output_values = result["adc_voltage"].to_numpy().flatten()
        np.testing.assert_array_equal(original_values, output_values)

    def test_payload_sensor_row_count_unchanged(
        self, basic_synchronizer, payload_sensor
    ):
        """Number of rows must equal the original sensor row count when common_time=False."""
        basic_synchronizer.add_payload_sensor("adc", payload_sensor)
        result = basic_synchronizer.synchronize(
            target_rate={"drone": 10.0, "payload": 10.0},
            common_time=False,
        )
        assert len(result["adc_voltage"]) == len(payload_sensor)

    def test_payload_timestamp_shifted_by_zero_when_no_offsets(
        self, basic_synchronizer, payload_sensor
    ):
        """When no inclinometer/litchi offsets exist the timestamp shift is zero."""
        basic_synchronizer.add_payload_sensor("adc", payload_sensor)
        result = basic_synchronizer.synchronize(
            target_rate={"drone": 10.0, "payload": 10.0},
            common_time=False,
        )
        original_ts = payload_sensor["timestamp"].to_numpy().flatten()
        output_ts = result["adc_timestamp"].to_numpy().flatten()
        # Total offset = inclinometer_offset (0.0) + litchi_offset (0.0) = 0.0
        np.testing.assert_allclose(output_ts, original_ts, atol=1e-9)

    def test_reference_gps_always_included(self, basic_synchronizer, payload_sensor):
        """reference_gps must be present regardless of common_time."""
        basic_synchronizer.add_payload_sensor("adc", payload_sensor)
        result = basic_synchronizer.synchronize(
            target_rate={"drone": 10.0, "payload": 10.0},
            common_time=False,
        )
        assert "reference_gps" in result
        assert isinstance(result["reference_gps"], pl.DataFrame)


# ---------------------------------------------------------------------------
# common_time=True vs False produce different shapes
# ---------------------------------------------------------------------------


class TestCommonTimeDifference:
    """Verify that common_time=True and False produce observably different output."""

    def test_shapes_differ_when_sensor_rate_differs_from_reference(
        self, reference_gps, payload_sensor
    ):
        """
        When the sensor native rate differs from target_rate the row count
        should differ between common_time=True and common_time=False.
        """
        # payload_sensor is at 10 Hz (101 rows for 10 s), but we request 3 Hz
        target_rate = 3.0

        sync_true = Synchronizer()
        sync_true.add_gps_reference(reference_gps)
        sync_true.add_payload_sensor("adc", payload_sensor)
        result_true = sync_true.synchronize(
            target_rate={"drone": 10.0, "payload": target_rate},
            common_time=True,
        )

        sync_false = Synchronizer()
        sync_false.add_gps_reference(reference_gps)
        sync_false.add_payload_sensor("adc", payload_sensor)
        result_false = sync_false.synchronize(
            target_rate={"drone": 10.0, "payload": target_rate},
            common_time=False,
        )

        n_true = len(result_true["adc_timestamp"])
        n_false = len(result_false["adc_timestamp"])
        # common_time=True: int(10 * 3) + 1 = 31 rows
        # common_time=False: same as input = 101 rows
        assert n_true != n_false
        assert (
            n_true
            == int(
                (reference_gps["timestamp"][-1] - reference_gps["timestamp"][0])
                * target_rate
            )
            + 1
        )
        assert n_false == len(payload_sensor)


# ---------------------------------------------------------------------------
# Flight.sync() – common_time parameter forwarded correctly
# ---------------------------------------------------------------------------


@pytest.fixture
def flight_with_gps():
    """Flight instance with GPS payload data pre-loaded."""
    flight_info = {
        "drone_data_folder_path": "/tmp/test_flight/drone",
        "aux_data_folder_path": "/tmp/test_flight/aux",
    }
    flight = Flight(flight_info)

    gps_data = pl.DataFrame(
        {
            "timestamp": np.linspace(0.0, 10.0, 11),
            "posllh_lat": 45.0 + 0.0001 * np.linspace(0.0, 10.0, 11),
            "posllh_lon": 10.0 + 0.0001 * np.linspace(0.0, 10.0, 11),
            "posllh_height": 100.0 + np.linspace(0.0, 10.0, 11),
        }
    )
    flight.raw_data.payload_data = PayloadData()
    flight.raw_data.payload_data.gps = gps_data

    # Also add a simple ADC sensor
    adc_data = pl.DataFrame(
        {
            "timestamp": np.linspace(0.0, 10.0, 101),
            "voltage": 3.3 + 0.01 * np.sin(np.linspace(0.0, 10.0, 101)),
        }
    )
    flight.raw_data.payload_data.adc = adc_data

    return flight


class TestFlightSyncCommonTime:
    """Tests for Flight.sync() common_time parameter."""

    def test_default_common_time_true(self, flight_with_gps):
        """Flight.sync() default (common_time=True) resamples to target_rate."""
        flight = flight_with_gps
        target_payload_rate = 5.0
        result = flight.sync(
            target_rate={"drone": 1.0, "payload": target_payload_rate},
            common_time=True,
        )
        assert isinstance(result, dict)
        assert "adc_timestamp" in result
        # Timestamps should be uniformly spaced
        ts = result["adc_timestamp"].to_numpy()
        diffs = np.diff(ts)
        assert np.allclose(diffs, diffs[0], rtol=1e-6)

    def test_common_time_false_preserves_original_rows(self, flight_with_gps):
        """Flight.sync(common_time=False) must not resample the sensor data."""
        flight = flight_with_gps
        original_n = len(flight.raw_data.payload_data.adc)
        result = flight.sync(
            target_rate={"drone": 1.0, "payload": 10.0},
            common_time=False,
        )
        assert "adc_voltage" in result
        assert len(result["adc_voltage"]) == original_n

    def test_common_time_false_preserves_values(self, flight_with_gps):
        """Flight.sync(common_time=False) must not change the sensor values."""
        flight = flight_with_gps
        original_values = (
            flight.raw_data.payload_data.adc["voltage"].to_numpy().flatten()
        )
        result = flight.sync(
            target_rate={"drone": 1.0, "payload": 10.0},
            common_time=False,
        )
        output_values = result["adc_voltage"].to_numpy().flatten()
        np.testing.assert_array_equal(original_values, output_values)

    def test_common_time_flag_stored_in_sync_data(self, flight_with_gps):
        """sync_data attribute is populated after calling sync()."""
        flight = flight_with_gps
        flight.sync(target_rate={"drone": 1.0, "payload": 10.0}, common_time=True)
        assert flight.sync_data is not None
        assert isinstance(flight.sync_data, dict)

    def test_sync_result_values_are_dataframes(self, flight_with_gps):
        """All values in sync_data must be pl.DataFrame for both common_time modes."""
        flight = flight_with_gps
        for flag in (True, False):
            # Re-create fixture state by resetting sync_data
            flight.sync_data = None
            result = flight.sync(
                target_rate={"drone": 1.0, "payload": 10.0},
                common_time=flag,
            )
            for key, value in result.items():
                assert isinstance(value, pl.DataFrame), (
                    f"Expected pl.DataFrame for key '{key}' with common_time={flag}"
                )
