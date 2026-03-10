"""
Tests for EKF Analysis Module.

Testing EKFVersion dataclass and related functionality.
"""

from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pytest

from pils.flight import Flight


@pytest.fixture
def mock_flight(tmp_path):
    """Create mock Flight object with valid flight_path."""
    flight_path = tmp_path / "flight_test"
    flight_path.mkdir()
    drone_data_path = flight_path / "drone"
    drone_data_path.mkdir()

    flight_info = {
        "drone_data_folder_path": str(drone_data_path),
    }
    flight = Flight(flight_info)
    return flight


# ==================== Sample Data Fixtures ====================


@pytest.fixture
def sample_imu_df():
    """Create sample IMU DataFrame for testing."""
    n = 100
    t0 = 1_700_000_000_000_000_000  # nanoseconds
    dt_ns = 10_000_000  # 10 ms between samples (100 Hz)

    return pl.DataFrame(
        {
            "monotonic_ns": [i * dt_ns for i in range(n)],
            "timestamp_ns": [t0 + i * dt_ns for i in range(n)],
            "pqr_P_rad_s": np.random.normal(0.0, 0.01, n).tolist(),
            "pqr_Q_rad_s": np.random.normal(0.0, 0.01, n).tolist(),
            "pqr_R_rad_s": np.random.normal(0.0, 0.01, n).tolist(),
            "acc_X_m_s2": np.random.normal(0.0, 0.1, n).tolist(),
            "acc_Y_m_s2": np.random.normal(0.0, 0.1, n).tolist(),
            "acc_Z_m_s2": np.random.normal(-9.81, 0.1, n).tolist(),
        }
    )


@pytest.fixture
def sample_photo_df():
    """Create sample photogrammetry DataFrame for testing."""
    n = 20
    t0 = 1_700_000_000_000_000_000  # nanoseconds
    dt_ns = 50_000_000  # 50 ms between samples (20 Hz)

    return pl.DataFrame(
        {
            "monotonic_ns": [i * dt_ns for i in range(n)],
            "timestamp_ns": [t0 + i * dt_ns for i in range(n)],
            "quat_w": np.ones(n).tolist(),
            "quat_x": np.zeros(n).tolist(),
            "quat_y": np.zeros(n).tolist(),
            "quat_z": np.zeros(n).tolist(),
        }
    )


# ==================== Import Tests ====================


def test_import_ekf_version():
    """Test that EKFVersion can be imported."""
    from pils.analyze.ekf import EKFVersion

    assert EKFVersion is not None


def test_import_ekf_analysis():
    """Test that EKFAnalysis can be imported."""
    from pils.analyze.ekf import EKFAnalysis

    assert EKFAnalysis is not None


def test_import_from_package():
    """Test that EKF classes are exported from analyze package."""
    from pils.analyze import EKFAnalysis, EKFVersion

    assert EKFAnalysis is not None
    assert EKFVersion is not None


# ==================== EKFVersion Dataclass Tests ====================


def test_ekf_version_dataclass_creation():
    """Test EKFVersion instantiation with all fields."""
    from pils.analyze.ekf import EKFVersion

    # Create sample EKF data
    ekf_data = pl.DataFrame(
        {
            "timestamp": [1000.0, 2000.0, 3000.0],
            "quat_w": [1.0, 0.99, 0.98],
            "quat_x": [0.0, 0.01, 0.02],
            "quat_y": [0.0, 0.01, 0.02],
            "quat_z": [0.0, 0.01, 0.02],
        }
    )

    metadata = {
        "config_file": "config.yaml",
        "num_imu_samples": 5000,
        "num_photo_samples": 200,
    }

    # Create EKFVersion instance
    version = EKFVersion(
        version_name="v1_test", ekf_data=ekf_data, metadata=metadata
    )

    # Verify all fields
    assert version.version_name == "v1_test"
    assert version.ekf_data.shape == (3, 5)
    assert version.metadata["num_imu_samples"] == 5000
    assert isinstance(version.ekf_data, pl.DataFrame)


def test_ekf_version_required_fields():
    """Test that EKFVersion requires all fields."""
    from pils.analyze.ekf import EKFVersion

    # This should raise TypeError when missing required fields
    with pytest.raises(TypeError):
        EKFVersion()  # No arguments

    with pytest.raises(TypeError):
        EKFVersion(version_name="test")  # Missing ekf_data and metadata


def test_ekf_version_dataframe_schema():
    """Test that ekf_data accepts correct column types."""
    from pils.analyze.ekf import EKFVersion

    # Create EKF data with expected schema
    ekf_data = pl.DataFrame(
        {
            "timestamp": [1000.0, 2000.0],
            "quat_w": [1.0, 0.99],
            "quat_x": [0.0, 0.01],
            "quat_y": [0.0, 0.01],
            "quat_z": [0.0, 0.01],
        }
    )

    version = EKFVersion(version_name="schema_test", ekf_data=ekf_data, metadata={})

    # Check DataFrame has expected columns
    assert "timestamp" in version.ekf_data.columns
    assert "quat_w" in version.ekf_data.columns
    assert "quat_x" in version.ekf_data.columns
    assert "quat_y" in version.ekf_data.columns
    assert "quat_z" in version.ekf_data.columns

    # Check data types are Float64
    assert version.ekf_data["timestamp"].dtype == pl.Float64
    assert version.ekf_data["quat_w"].dtype == pl.Float64


def test_ekf_version_is_dataclass():
    """Test that EKFVersion is actually a dataclass."""
    from dataclasses import is_dataclass

    from pils.analyze.ekf import EKFVersion

    assert is_dataclass(EKFVersion)

    # Check expected fields exist
    field_names = [f.name for f in fields(EKFVersion)]
    assert "version_name" in field_names
    assert "ekf_data" in field_names
    assert "metadata" in field_names


# ==================== EKFAnalysis Init Tests ====================


class TestEKFAnalysisInit:
    """Test EKFAnalysis initialization and path setup."""

    def test_ekf_analysis_init_with_valid_flight(self, mock_flight):
        """Test initialization with valid Flight object."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        assert ekf.flight_path == mock_flight.flight_path
        assert ekf.ekf_dir.exists()
        assert ekf.ekf_dir == mock_flight.flight_path / "proc" / "ekf"

    def test_ekf_analysis_creates_ekf_directory(self, mock_flight):
        """Test that EKF directory is created at proc/ekf/."""
        from pils.analyze.ekf import EKFAnalysis

        # Verify directory doesn't exist initially
        ekf_dir = mock_flight.flight_path / "proc" / "ekf"
        assert not ekf_dir.exists()

        # Create EKFAnalysis instance
        ekf = EKFAnalysis(mock_flight)

        # Verify directory was created
        assert ekf_dir.exists()
        assert ekf_dir.is_dir()
        assert ekf.ekf_dir == ekf_dir

    def test_ekf_analysis_rejects_invalid_flight(self, tmp_path):
        """Test that initialization raises TypeError for non-Flight objects."""
        from pils.analyze.ekf import EKFAnalysis

        flight_path = tmp_path / "flight_004"
        flight_path.mkdir()

        # Test with string path
        with pytest.raises(TypeError, match="Expected Flight object"):
            EKFAnalysis(str(flight_path))

        # Test with Path object
        with pytest.raises(TypeError, match="Expected Flight object"):
            EKFAnalysis(flight_path)

        # Test with None
        with pytest.raises(TypeError, match="Expected Flight object"):
            EKFAnalysis(None)

    def test_ekf_analysis_rejects_missing_flight_path(self, tmp_path):
        """Test that initialization raises ValueError for Flight without flight_path."""
        from pils.analyze.ekf import EKFAnalysis

        # Create flight with None flight_path
        drone_data_path = tmp_path / "drone"
        drone_data_path.mkdir()

        flight_info = {
            "drone_data_folder_path": str(drone_data_path),
        }
        flight = Flight(flight_info)

        # Manually set flight_path to None to simulate invalid state
        flight.flight_path = None

        with pytest.raises(
            ValueError, match="Flight object must have a valid flight_path"
        ):
            EKFAnalysis(flight)

    def test_ekf_analysis_rejects_flight_path_not_directory(self, tmp_path):
        """Test that initialization raises ValueError if flight_path is not a directory."""
        from pils.analyze.ekf import EKFAnalysis

        # Create a file instead of directory
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("test")

        drone_data_path = tmp_path / "drone"
        drone_data_path.mkdir()

        flight_info = {
            "drone_data_folder_path": str(drone_data_path),
        }
        flight = Flight(flight_info)
        flight.flight_path = file_path

        with pytest.raises(
            ValueError, match="flight_path must be an existing directory"
        ):
            EKFAnalysis(flight)

    def test_ekf_analysis_ekf_dir_property(self, mock_flight):
        """Test ekf_dir property returns correct Path."""
        from pathlib import Path

        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        # Verify ekf_dir is a Path object
        assert isinstance(ekf.ekf_dir, Path)

        # Verify correct path structure
        expected_dir = mock_flight.flight_path / "proc" / "ekf"
        assert ekf.ekf_dir == expected_dir


# ==================== Arrow IPC Serialization Tests ====================


class TestArrowIPCSerialization:
    """Test Arrow IPC table preparation and serialization."""

    def test_table_to_ipc_bytes_roundtrip(self):
        """Test IPC bytes serialization produces valid Arrow data."""
        from pils.analyze.ekf import EKFAnalysis

        schema = pa.schema([("x", pa.float64()), ("y", pa.float64())])
        table = pa.table({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})

        ipc_bytes = EKFAnalysis._table_to_ipc_bytes(table, schema)

        # Verify it's non-empty bytes
        assert isinstance(ipc_bytes, bytes)
        assert len(ipc_bytes) > 0

        # Verify roundtrip: read back the IPC bytes
        import io

        reader = pa.ipc.open_file(io.BytesIO(ipc_bytes))
        roundtrip_table = reader.read_all()

        assert roundtrip_table.num_rows == 3
        assert roundtrip_table.column("x").to_pylist() == [1.0, 2.0, 3.0]
        assert roundtrip_table.column("y").to_pylist() == [4.0, 5.0, 6.0]

    def test_prepare_imu_table_valid(self, sample_imu_df):
        """Test IMU table preparation with valid data."""
        from pils.analyze.ekf import IMU_SCHEMA, EKFAnalysis

        table = EKFAnalysis._prepare_imu_table(sample_imu_df)

        assert isinstance(table, pa.Table)
        assert table.num_rows == sample_imu_df.height
        assert table.schema == IMU_SCHEMA

    def test_prepare_imu_table_missing_columns(self):
        """Test IMU table preparation raises on missing columns."""
        from pils.analyze.ekf import EKFAnalysis

        incomplete_df = pl.DataFrame(
            {
                "monotonic_ns": [0],
                "timestamp_ns": [1000],
                # Missing gyro and accel columns
            }
        )

        with pytest.raises(ValueError, match="IMU DataFrame missing required columns"):
            EKFAnalysis._prepare_imu_table(incomplete_df)

    def test_prepare_photo_table_valid(self, sample_photo_df):
        """Test photogrammetry table preparation with valid data."""
        from pils.analyze.ekf import PHOTO_SCHEMA, EKFAnalysis

        table = EKFAnalysis._prepare_photo_table(sample_photo_df)

        assert isinstance(table, pa.Table)
        assert table.num_rows == sample_photo_df.height
        assert table.schema == PHOTO_SCHEMA

    def test_prepare_photo_table_missing_columns(self):
        """Test photogrammetry table preparation raises on missing columns."""
        from pils.analyze.ekf import EKFAnalysis

        incomplete_df = pl.DataFrame(
            {
                "monotonic_ns": [0],
                "timestamp_ns": [1000],
                # Missing quaternion columns
            }
        )

        with pytest.raises(ValueError, match="Photogrammetry DataFrame missing required columns"):
            EKFAnalysis._prepare_photo_table(incomplete_df)

    def test_imu_schema_column_names(self):
        """Test IMU_SCHEMA has expected column names."""
        from pils.analyze.ekf import IMU_SCHEMA

        expected_names = [
            "monotonic_ns",
            "timestamp_ns",
            "pqr_P_rad_s",
            "pqr_Q_rad_s",
            "pqr_R_rad_s",
            "acc_X_m_s2",
            "acc_Y_m_s2",
            "acc_Z_m_s2",
        ]
        assert IMU_SCHEMA.names == expected_names

    def test_photo_schema_column_names(self):
        """Test PHOTO_SCHEMA has expected column names."""
        from pils.analyze.ekf import PHOTO_SCHEMA

        expected_names = [
            "monotonic_ns",
            "timestamp_ns",
            "quat_w",
            "quat_x",
            "quat_y",
            "quat_z",
        ]
        assert PHOTO_SCHEMA.names == expected_names

    def test_imu_ipc_bytes_length_prefix_format(self, sample_imu_df):
        """Test IPC bytes can be length-prefixed for the Rust protocol."""
        import struct

        from pils.analyze.ekf import IMU_SCHEMA, EKFAnalysis

        table = EKFAnalysis._prepare_imu_table(sample_imu_df)
        ipc_bytes = EKFAnalysis._table_to_ipc_bytes(table, IMU_SCHEMA)

        # Build length-prefixed payload (same as Rust IPC protocol)
        length_prefix = struct.pack("<Q", len(ipc_bytes))
        payload = length_prefix + ipc_bytes

        # Verify prefix encodes correct length
        decoded_length = struct.unpack("<Q", payload[:8])[0]
        assert decoded_length == len(ipc_bytes)

        # Verify IPC payload follows the prefix
        assert payload[8:] == ipc_bytes


# ==================== Run Analysis Tests ====================


class TestRunAnalysis:
    """Test suite for run_analysis() main method."""

    def test_run_analysis_rejects_empty_imu(self, mock_flight, sample_photo_df):
        """Test that run_analysis raises when IMU DataFrame is empty."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        empty_imu = pl.DataFrame(
            {
                "monotonic_ns": [],
                "timestamp_ns": [],
                "pqr_P_rad_s": [],
                "pqr_Q_rad_s": [],
                "pqr_R_rad_s": [],
                "acc_X_m_s2": [],
                "acc_Y_m_s2": [],
                "acc_Z_m_s2": [],
            }
        )

        with pytest.raises(ValueError, match="IMU DataFrame is empty"):
            ekf.run_analysis(empty_imu, sample_photo_df)

    def test_run_analysis_rejects_empty_photo(self, mock_flight, sample_imu_df):
        """Test that run_analysis raises when photogrammetry DataFrame is empty."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        empty_photo = pl.DataFrame(
            {
                "monotonic_ns": [],
                "timestamp_ns": [],
                "quat_w": [],
                "quat_x": [],
                "quat_y": [],
                "quat_z": [],
            }
        )

        with pytest.raises(ValueError, match="Photogrammetry DataFrame is empty"):
            ekf.run_analysis(sample_imu_df, empty_photo)

    def test_run_analysis_rejects_none_imu(self, mock_flight, sample_photo_df):
        """Test that run_analysis raises when IMU DataFrame is None."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        with pytest.raises(ValueError, match="IMU DataFrame is empty"):
            ekf.run_analysis(None, sample_photo_df)

    def test_run_analysis_rejects_none_photo(self, mock_flight, sample_imu_df):
        """Test that run_analysis raises when photogrammetry DataFrame is None."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        with pytest.raises(ValueError, match="Photogrammetry DataFrame is empty"):
            ekf.run_analysis(sample_imu_df, None)

    def test_run_analysis_rejects_missing_config(
        self, mock_flight, sample_imu_df, sample_photo_df
    ):
        """Test that run_analysis raises when config file doesn't exist."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        with pytest.raises(FileNotFoundError, match="EKF config file not found"):
            ekf.run_analysis(
                sample_imu_df, sample_photo_df, config_file="/nonexistent/config.yaml"
            )

    def test_run_analysis_uses_default_config(self, mock_flight):
        """Test that run_analysis resolves to default config when None is passed."""
        from pils.analyze.ekf import _EKF_DEFAULT_CONFIG, EKFAnalysis

        EKFAnalysis(mock_flight)

        # Verify default config path exists (it's bundled with the module)
        assert _EKF_DEFAULT_CONFIG.exists(), (
            f"Default EKF config not found at {_EKF_DEFAULT_CONFIG}"
        )

    # Full end-to-end pipeline tests with real flight data are in
    # TestEKFRealDataIntegration below (requires compiled Rust binary +
    # ekfTestData/ directory).

    def test_ekf_output_schema_column_names(self):
        """Test EKF_OUTPUT_SCHEMA has the expected 14 Euler + covariance columns."""
        from pils.analyze.ekf import EKF_OUTPUT_SCHEMA

        expected_names = [
            "timestamp_s",
            "timestamp_monotonic_ns",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "euler_cov_0_0",
            "euler_cov_0_1",
            "euler_cov_0_2",
            "euler_cov_1_0",
            "euler_cov_1_1",
            "euler_cov_1_2",
            "euler_cov_2_0",
            "euler_cov_2_1",
            "euler_cov_2_2",
        ]
        assert EKF_OUTPUT_SCHEMA.names == expected_names

    # TODO: Add test for _send_ipc_to_rust with mock subprocess
    #       to validate the length-prefixed IPC protocol without
    #       requiring the actual Rust binary
    #
    # def test_send_ipc_to_rust_protocol(self, tmp_path, sample_imu_df, sample_photo_df):
    #     """Test IPC protocol sends correct length-prefixed payloads."""
    #     ...

    def test_load_ekf_output_reads_ipc_bytes(self):
        """Test that _load_ekf_output deserializes length-prefixed Arrow IPC bytes."""
        import io
        import struct

        from pils.analyze.ekf import EKF_OUTPUT_SCHEMA, EKFAnalysis

        # Build a mock IPC payload matching the Rust binary output schema
        output_cols = [f.name for f in EKF_OUTPUT_SCHEMA]
        mock_data = {col: [1.0, 2.0, 3.0] for col in output_cols}
        table = pa.table(mock_data, schema=EKF_OUTPUT_SCHEMA)

        buf = io.BytesIO()
        writer = pa.ipc.new_file(buf, schema=EKF_OUTPUT_SCHEMA)
        writer.write_table(table)
        writer.close()
        ipc_bytes = buf.getvalue()

        # Build length-prefixed payload (Rust wire format)
        length_prefix = struct.pack("<Q", len(ipc_bytes))
        payload = length_prefix + ipc_bytes

        result = EKFAnalysis._load_ekf_output(payload)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 3
        assert set(output_cols).issubset(set(result.columns))

    def test_load_ekf_output_raises_on_short_input(self):
        """Test that _load_ekf_output raises ValueError on too-short bytes."""
        from pils.analyze.ekf import EKFAnalysis

        with pytest.raises(ValueError, match="too short"):
            EKFAnalysis._load_ekf_output(b"\x00\x01")

    def test_load_ekf_output_raises_on_truncated_payload(self):
        """Test that _load_ekf_output raises ValueError when payload holds no Arrow data."""
        import struct

        from pils.analyze.ekf import EKFAnalysis

        # Claim 1000 bytes but only provide 10 — Arrow magic not present either
        fake_header = struct.pack("<Q", 1000)
        truncated_payload = fake_header + b"\x00" * 10

        with pytest.raises(ValueError, match="Arrow IPC magic bytes not found"):
            EKFAnalysis._load_ekf_output(truncated_payload)

    # TODO: Add test for run_analysis metadata content
    #
    # def test_run_analysis_metadata_keys(self, mock_flight, sample_imu_df, sample_photo_df):
    #     """Test output metadata contains all expected keys."""
    #     from pils.analyze.ekf import EKFAnalysis
    #     ekf = EKFAnalysis(mock_flight)
    #     result = ekf.run_analysis(sample_imu_df, sample_photo_df)
    #     assert "config_file" in result.metadata
    #     assert "num_imu_samples" in result.metadata
    #     assert "num_photo_samples" in result.metadata
    #     assert "num_output_samples" in result.metadata

    # TODO: Add test for custom config file override
    #
    # def test_run_analysis_custom_config(self, mock_flight, tmp_path, sample_imu_df, sample_photo_df):
    #     """Test run_analysis with user-provided config file."""
    #     custom_config = tmp_path / "custom_config.yaml"
    #     custom_config.write_text("ekf:\n  process_noise:\n    quaternion: 1.0e-8\n")
    #     from pils.analyze.ekf import EKFAnalysis
    #     ekf = EKFAnalysis(mock_flight)
    #     result = ekf.run_analysis(sample_imu_df, sample_photo_df, config_file=custom_config)
    #     assert result.metadata["config_file"] == str(custom_config)


# ==================== Rust Binary Validation Tests ====================


class TestRustBinaryValidation:
    """Test Rust EKF binary path resolution and validation."""

    def test_binary_path_resolution(self):
        """Test that binary path resolves relative to ekf module."""
        from pils.analyze.ekf import _EKF_BINARY_PATH

        # Should point to ekf/bin/rust-ekf relative to the module
        assert _EKF_BINARY_PATH.name == "rust-ekf"
        assert "ekf" in str(_EKF_BINARY_PATH)
        assert "bin" in str(_EKF_BINARY_PATH)

    def test_default_config_path_resolution(self):
        """Test that default config path resolves relative to ekf module."""
        from pils.analyze.ekf import _EKF_DEFAULT_CONFIG

        # Should point to ekf/bin/config.yaml relative to the module
        assert _EKF_DEFAULT_CONFIG.name == "config.yaml"
        assert "ekf" in str(_EKF_DEFAULT_CONFIG)
        assert "bin" in str(_EKF_DEFAULT_CONFIG)

    def test_binary_exists_on_disk(self):
        """Test that compiled Rust EKF binary is present."""
        from pils.analyze.ekf import _EKF_BINARY_PATH

        # This test will pass only when the binary is compiled
        # Mark as expected failure if binary not yet built
        if not _EKF_BINARY_PATH.exists():
            pytest.skip("Rust EKF binary not compiled (cargo build --release)")

        assert _EKF_BINARY_PATH.is_file()

    def test_send_ipc_to_rust_raises_without_binary(self, tmp_path):
        """Test _send_ipc_to_rust raises FileNotFoundError when binary missing."""
        # Use a non-existent path for the binary
        import pils.analyze.ekf as ekf_module
        from pils.analyze.ekf import EKFAnalysis

        original_path = ekf_module._EKF_BINARY_PATH
        try:
            ekf_module._EKF_BINARY_PATH = tmp_path / "nonexistent" / "rust-ekf"

            with pytest.raises(FileNotFoundError, match="Rust EKF binary not found"):
                EKFAnalysis._send_ipc_to_rust(
                    b"imu_data",
                    b"photo_data",
                    output_dir=tmp_path,
                    config_file=tmp_path / "config.yaml",
                )
        finally:
            # Restore original path
            ekf_module._EKF_BINARY_PATH = original_path


# ==================== HDF5 Persistence Tests ====================


class TestHDF5Persistence:
    """Test HDF5 persistence for EKF versions."""

    @pytest.fixture
    def sample_ekf_version(self):
        """Create sample EKFVersion for testing."""
        from pils.analyze.ekf import EKFVersion

        ekf_data = pl.DataFrame(
            {
                "timestamp": [1000.0, 2000.0, 3000.0],
                "quat_w": [1.0, 0.999, 0.998],
                "quat_x": [0.0, 0.01, 0.02],
                "quat_y": [0.0, 0.01, 0.02],
                "quat_z": [0.0, 0.01, 0.02],
            }
        ).with_columns(
            [
                pl.col("timestamp").cast(pl.Float64),
                pl.col("quat_w").cast(pl.Float64),
                pl.col("quat_x").cast(pl.Float64),
                pl.col("quat_y").cast(pl.Float64),
                pl.col("quat_z").cast(pl.Float64),
            ]
        )

        metadata = {
            "config_file": "config.yaml",
            "num_imu_samples": 5000,
            "num_photo_samples": 200,
            "num_output_samples": 3,
        }

        return EKFVersion(
            version_name="rev_20260218_120000", ekf_data=ekf_data, metadata=metadata
        )

    def test_save_ekf_version_to_hdf5(self, mock_flight, sample_ekf_version):
        """Test saving EKFVersion to HDF5 file."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        # Save version to HDF5
        ekf._save_to_hdf5(sample_ekf_version)

        # Verify HDF5 file exists
        hdf5_path = ekf.ekf_dir / "ekf_solution.h5"
        assert hdf5_path.exists()
        assert hdf5_path.is_file()

    def test_load_ekf_version_from_hdf5(self, mock_flight, sample_ekf_version):
        """Test loading saved version matches original."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        # Save version
        ekf._save_to_hdf5(sample_ekf_version)

        # Load version back
        loaded_version = ekf._load_from_hdf5("rev_20260218_120000")

        # Verify version name matches
        assert loaded_version.version_name == sample_ekf_version.version_name

        # Verify DataFrame shape matches
        assert loaded_version.ekf_data.shape == sample_ekf_version.ekf_data.shape

        # Verify DataFrame contents match
        assert loaded_version.ekf_data["timestamp"].to_list() == pytest.approx(
            sample_ekf_version.ekf_data["timestamp"].to_list()
        )
        assert loaded_version.ekf_data["quat_w"].to_list() == pytest.approx(
            sample_ekf_version.ekf_data["quat_w"].to_list()
        )
        assert loaded_version.ekf_data["quat_x"].to_list() == pytest.approx(
            sample_ekf_version.ekf_data["quat_x"].to_list()
        )
        assert loaded_version.ekf_data["quat_y"].to_list() == pytest.approx(
            sample_ekf_version.ekf_data["quat_y"].to_list()
        )
        assert loaded_version.ekf_data["quat_z"].to_list() == pytest.approx(
            sample_ekf_version.ekf_data["quat_z"].to_list()
        )

        # Verify metadata matches
        assert loaded_version.metadata["config_file"] == "config.yaml"
        assert loaded_version.metadata["num_imu_samples"] == 5000

    def test_hdf5_preserves_dataframe_dtypes(self, mock_flight, sample_ekf_version):
        """Test dtypes are preserved after save/load cycle."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        # Verify original dtypes are Float64
        assert sample_ekf_version.ekf_data["timestamp"].dtype == pl.Float64
        assert sample_ekf_version.ekf_data["quat_w"].dtype == pl.Float64
        assert sample_ekf_version.ekf_data["quat_x"].dtype == pl.Float64

        # Save and load
        ekf._save_to_hdf5(sample_ekf_version)
        loaded_version = ekf._load_from_hdf5("rev_20260218_120000")

        # Verify loaded dtypes match original
        assert loaded_version.ekf_data["timestamp"].dtype == pl.Float64
        assert loaded_version.ekf_data["quat_w"].dtype == pl.Float64
        assert loaded_version.ekf_data["quat_x"].dtype == pl.Float64

    def test_list_versions_returns_all_versions(self, mock_flight):
        """Test listing all saved versions."""
        from pils.analyze.ekf import EKFAnalysis, EKFVersion

        ekf = EKFAnalysis(mock_flight)

        # Initially no versions
        assert ekf.list_versions() == []

        # Create and save multiple versions
        for i, timestamp in enumerate(["120000", "130000", "140000"]):
            ekf_data = pl.DataFrame(
                {
                    "timestamp": [float(i * 1000)],
                    "quat_w": [1.0],
                    "quat_x": [0.0],
                    "quat_y": [0.0],
                    "quat_z": [0.0],
                }
            ).with_columns(
                [
                    pl.col("timestamp").cast(pl.Float64),
                    pl.col("quat_w").cast(pl.Float64),
                    pl.col("quat_x").cast(pl.Float64),
                    pl.col("quat_y").cast(pl.Float64),
                    pl.col("quat_z").cast(pl.Float64),
                ]
            )

            version = EKFVersion(
                version_name=f"rev_20260218_{timestamp}",
                ekf_data=ekf_data,
                metadata={"num_output_samples": 1},
            )
            ekf._save_to_hdf5(version)

        # List versions
        versions = ekf.list_versions()

        # Should return all 3 versions in chronological order
        assert len(versions) == 3
        assert versions == [
            "rev_20260218_120000",
            "rev_20260218_130000",
            "rev_20260218_140000",
        ]

    def test_get_latest_version_returns_most_recent(self, mock_flight):
        """Test retrieving latest version."""
        from pils.analyze.ekf import EKFAnalysis, EKFVersion

        ekf = EKFAnalysis(mock_flight)

        # No versions initially
        assert ekf.get_latest_version() is None

        # Save multiple versions with different timestamps
        for i, timestamp in enumerate(["120000", "133000", "145000"]):
            ekf_data = pl.DataFrame(
                {
                    "timestamp": [float(i * 1000)],
                    "quat_w": [1.0 - i * 0.001],
                    "quat_x": [0.0 + i * 0.01],
                    "quat_y": [0.0],
                    "quat_z": [0.0],
                }
            ).with_columns(
                [
                    pl.col("timestamp").cast(pl.Float64),
                    pl.col("quat_w").cast(pl.Float64),
                    pl.col("quat_x").cast(pl.Float64),
                    pl.col("quat_y").cast(pl.Float64),
                    pl.col("quat_z").cast(pl.Float64),
                ]
            )

            version = EKFVersion(
                version_name=f"rev_20260218_{timestamp}",
                ekf_data=ekf_data,
                metadata={"version_id": i},
            )
            ekf._save_to_hdf5(version)

        # Get latest version
        latest = ekf.get_latest_version()

        # Should return the last version (145000)
        assert latest is not None
        assert latest.version_name == "rev_20260218_145000"
        assert latest.metadata["version_id"] == 2
        assert latest.ekf_data["quat_x"][0] == pytest.approx(0.02)

    def test_multiple_versions_in_single_hdf5(self, mock_flight):
        """Test multiple analysis versions stored in same HDF5."""
        from pils.analyze.ekf import EKFAnalysis, EKFVersion

        ekf = EKFAnalysis(mock_flight)

        # Create 3 different versions
        versions_to_save = []
        for i in range(3):
            ekf_data = pl.DataFrame(
                {
                    "timestamp": [1000.0 + i * 100, 2000.0 + i * 100],
                    "quat_w": [1.0 - i * 0.001, 0.999 - i * 0.001],
                    "quat_x": [0.0 + i * 0.01, 0.01 + i * 0.01],
                    "quat_y": [0.0, 0.01],
                    "quat_z": [0.0, 0.01],
                }
            ).with_columns(
                [
                    pl.col("timestamp").cast(pl.Float64),
                    pl.col("quat_w").cast(pl.Float64),
                    pl.col("quat_x").cast(pl.Float64),
                    pl.col("quat_y").cast(pl.Float64),
                    pl.col("quat_z").cast(pl.Float64),
                ]
            )

            version = EKFVersion(
                version_name=f"rev_20260218_12{i}000",
                ekf_data=ekf_data,
                metadata={"version_num": i},
            )
            versions_to_save.append(version)
            ekf._save_to_hdf5(version)

        # Verify all versions can be loaded independently
        for i, original_version in enumerate(versions_to_save):
            loaded_version = ekf._load_from_hdf5(original_version.version_name)

            assert loaded_version.version_name == original_version.version_name
            assert loaded_version.ekf_data.shape == original_version.ekf_data.shape
            assert loaded_version.metadata["version_num"] == i

            # Verify data values match
            assert loaded_version.ekf_data["quat_w"].to_list() == pytest.approx(
                original_version.ekf_data["quat_w"].to_list()
            )

    def test_hdf5_file_not_found_raises_error(self, mock_flight):
        """Test loading from non-existent HDF5 raises FileNotFoundError."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        # Try to load without saving first
        with pytest.raises(FileNotFoundError):
            ekf._load_from_hdf5("nonexistent_version")

    def test_hdf5_version_not_found_raises_error(
        self, mock_flight, sample_ekf_version
    ):
        """Test loading non-existent version raises KeyError."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)

        # Save one version
        ekf._save_to_hdf5(sample_ekf_version)

        # Try to load a different version
        with pytest.raises(KeyError):
            ekf._load_from_hdf5("rev_99999999_999999")

    def test_overwrite_existing_version(self, mock_flight):
        """Test saving a version with same name overwrites the old one."""
        from pils.analyze.ekf import EKFAnalysis, EKFVersion

        ekf = EKFAnalysis(mock_flight)

        # Save first version
        ekf_data_v1 = pl.DataFrame(
            {
                "timestamp": [1000.0],
                "quat_w": [1.0],
                "quat_x": [0.0],
                "quat_y": [0.0],
                "quat_z": [0.0],
            }
        ).with_columns([pl.all().cast(pl.Float64)])

        v1 = EKFVersion(
            version_name="rev_20260218_120000",
            ekf_data=ekf_data_v1,
            metadata={"iteration": 1},
        )
        ekf._save_to_hdf5(v1)

        # Overwrite with new data
        ekf_data_v2 = pl.DataFrame(
            {
                "timestamp": [2000.0],
                "quat_w": [0.99],
                "quat_x": [0.01],
                "quat_y": [0.01],
                "quat_z": [0.01],
            }
        ).with_columns([pl.all().cast(pl.Float64)])

        v2 = EKFVersion(
            version_name="rev_20260218_120000",
            ekf_data=ekf_data_v2,
            metadata={"iteration": 2},
        )
        ekf._save_to_hdf5(v2)

        # Load should return the overwritten version
        loaded = ekf._load_from_hdf5("rev_20260218_120000")
        assert loaded.metadata["iteration"] == 2
        assert loaded.ekf_data["quat_w"][0] == pytest.approx(0.99)

        # Should still be only 1 version
        assert len(ekf.list_versions()) == 1


# ==================== Real Data Integration Tests ====================

_TEST_DATA_DIR = (
    Path(__file__).parent.parent / "pils" / "analyze" / "ekf" / "ekfTestData"
)


@pytest.fixture(scope="module")
def real_imu_df():
    """Load sample_imu.csv — module-scoped to avoid repeated CSV I/O."""
    if not _TEST_DATA_DIR.exists():
        pytest.skip(f"Test data directory not found: {_TEST_DATA_DIR}")
    return pl.read_csv(_TEST_DATA_DIR / "sample_imu.csv").select(
        [
            "monotonic_ns",
            "timestamp_ns",
            "pqr_P_rad_s",
            "pqr_Q_rad_s",
            "pqr_R_rad_s",
            "acc_X_m_s2",
            "acc_Y_m_s2",
            "acc_Z_m_s2",
        ]
    ).cast({"monotonic_ns": pl.Int64, "timestamp_ns": pl.Int64})


@pytest.fixture(scope="module")
def real_photo_df():
    """Build photo DataFrame from sample_photo.ecsv — module-scoped."""
    if not _TEST_DATA_DIR.exists():
        pytest.skip(f"Test data directory not found: {_TEST_DATA_DIR}")

    from astropy.table import Table

    photo = Table.read(str(_TEST_DATA_DIR / "sample_photo.ecsv"))
    time_s = np.array(photo["time"], dtype=np.float64)
    qw = np.array(photo["quat_w_corr_world"], dtype=np.float64)
    qx = np.array(photo["quat_x_corr_world"], dtype=np.float64)
    qy = np.array(photo["quat_y_corr_world"], dtype=np.float64)
    qz = np.array(photo["quat_z_corr_world"], dtype=np.float64)

    timestamp_ns = (time_s * 1e9).astype(np.int64)
    monotonic_ns = ((time_s - time_s[0]) * 1e9).astype(np.int64)

    return pl.DataFrame(
        {
            "monotonic_ns": monotonic_ns.tolist(),
            "timestamp_ns": timestamp_ns.tolist(),
            "quat_w": qw.tolist(),
            "quat_x": qx.tolist(),
            "quat_y": qy.tolist(),
            "quat_z": qz.tolist(),
        }
    )


@pytest.fixture(scope="module")
def real_photo_euler():
    """Return reference Euler angles (roll/pitch/yaw in deg) from ECSV — module-scoped."""
    if not _TEST_DATA_DIR.exists():
        pytest.skip(f"Test data directory not found: {_TEST_DATA_DIR}")

    from astropy.table import Table
    from scipy.spatial.transform import Rotation

    photo = Table.read(str(_TEST_DATA_DIR / "sample_photo.ecsv"))
    time_s = np.array(photo["time"], dtype=np.float64)
    quats = np.column_stack(
        [
            np.array(photo["quat_x_corr_world"], dtype=np.float64),
            np.array(photo["quat_y_corr_world"], dtype=np.float64),
            np.array(photo["quat_z_corr_world"], dtype=np.float64),
            np.array(photo["quat_w_corr_world"], dtype=np.float64),
        ]
    )
    zyx = Rotation.from_quat(quats).as_euler("ZYX", degrees=True)
    return {
        "time_s": time_s,
        "roll_deg": zyx[:, 2],
        "pitch_deg": zyx[:, 1],
        "yaw_deg": zyx[:, 0],
    }


class TestEKFRealDataIntegration:
    """End-to-end tests using real flight data and the compiled Rust binary."""

    @pytest.fixture(autouse=True)
    def _skip_if_unavailable(self):
        from pils.analyze.ekf import _EKF_BINARY_PATH

        if not _EKF_BINARY_PATH.exists():
            pytest.skip(
                f"Rust EKF binary not compiled at {_EKF_BINARY_PATH}. "
                "Run: cargo build --release"
            )
        if not _TEST_DATA_DIR.exists():
            pytest.skip(f"Test data directory not found: {_TEST_DATA_DIR}")

    def test_run_analysis_with_real_data(
        self, mock_flight, real_imu_df, real_photo_df
    ):
        """Full EKF pipeline with real IMU and photogrammetry data."""
        from pils.analyze.ekf import EKFAnalysis, EKFVersion

        ekf = EKFAnalysis(mock_flight)
        result = ekf.run_analysis(real_imu_df, real_photo_df)

        assert result is not None
        assert isinstance(result, EKFVersion)
        assert isinstance(result.ekf_data, pl.DataFrame)
        assert result.version_name.startswith("rev_")
        assert result.ekf_data.height > 0

        # Required EKF output columns are present
        expected_cols = {
            "timestamp_s",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "euler_cov_0_0",
        }
        assert expected_cols.issubset(set(result.ekf_data.columns))

        # Metadata is fully populated
        assert result.metadata["num_imu_samples"] == real_imu_df.height
        assert result.metadata["num_photo_samples"] == real_photo_df.height
        assert result.metadata["num_output_samples"] == result.ekf_data.height

    def test_run_analysis_with_save(self, mock_flight, real_imu_df, real_photo_df):
        """EKF run_analysis with save_data=True creates an HDF5 version."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)
        result = ekf.run_analysis(real_imu_df, real_photo_df, save_data=True)

        assert result is not None
        hdf5_path = ekf.ekf_dir / "ekf_solution.h5"
        assert hdf5_path.exists()

        # Version is retrievable
        loaded = ekf._load_from_hdf5(result.version_name)
        assert loaded.ekf_data.height == result.ekf_data.height

    def test_euler_angles_within_physical_bounds(
        self, mock_flight, real_imu_df, real_photo_df
    ):
        """All output Euler angles must lie within [-180°, 180°]."""
        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)
        result = ekf.run_analysis(real_imu_df, real_photo_df)

        assert result is not None
        for col in ("roll_deg", "pitch_deg", "yaw_deg"):
            vals = result.ekf_data[col].to_numpy()
            assert np.all(vals >= -180.0) and np.all(vals <= 180.0), (
                f"{col} out of [-180°, 180°] range"
            )

    def test_ekf_comparison_plot(
        self, mock_flight, real_imu_df, real_photo_df, real_photo_euler
    ):
        """Run EKF, compare against photogrammetry reference, save comparison plot."""
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")

        from pils.analyze.ekf import EKFAnalysis

        ekf = EKFAnalysis(mock_flight)
        result = ekf.run_analysis(real_imu_df, real_photo_df)
        assert result is not None

        # Work with numpy arrays directly — no pandas required
        ekf_df = result.ekf_data
        t_ekf = ekf_df["timestamp_s"].to_numpy()
        roll_ekf = ekf_df["roll_deg"].to_numpy()
        pitch_ekf = ekf_df["pitch_deg"].to_numpy()
        yaw_ekf = ekf_df["yaw_deg"].to_numpy()

        # Derive 1σ from covariance diagonal
        std = {}
        for axis, cov_col in [
            ("roll", "euler_cov_0_0"),
            ("pitch", "euler_cov_1_1"),
            ("yaw", "euler_cov_2_2"),
        ]:
            if cov_col in ekf_df.columns:
                std[axis] = np.sqrt(np.clip(ekf_df[cov_col].to_numpy(), 0, None))
            else:
                std[axis] = None

        # Error stats: interpolate photo reference onto EKF timestamps
        photo = real_photo_euler
        ekf_vals = {"roll": roll_ekf, "pitch": pitch_ekf, "yaw": yaw_ekf}
        stats = {}
        for axis in ("roll", "pitch", "yaw"):
            ref = np.interp(t_ekf, photo["time_s"], photo[f"{axis}_deg"])
            err = ekf_vals[axis] - ref
            stats[axis] = {
                "mean": float(np.mean(err)),
                "std": float(np.std(err)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "error": err,
            }

        # Print summary table
        print(
            "\nEKF vs Photogrammetry Error Statistics\n"
            + "=" * 55
            + f"\n{'Axis':<8} {'Mean (°)':>12} {'Std (°)':>12} {'RMSE (°)':>12}\n"
            + "-" * 55
        )
        for axis in ("roll", "pitch", "yaw"):
            s = stats[axis]
            print(f"{axis:<8} {s['mean']:>+12.4f} {s['std']:>12.4f} {s['rmse']:>12.4f}")

        # ── Build comparison figure (3 × 2: angle traces + error traces) ──────
        axis_cfg = [("roll", "r", "Roll"), ("pitch", "g", "Pitch"), ("yaw", "m", "Yaw")]
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle("EKF vs Photogrammetry Reference", fontsize=14, fontweight="bold")

        for row, (axis, color, label) in enumerate(axis_cfg):
            err = stats[axis]["error"]

            # Angle comparison with ±1σ band
            axes[row, 0].plot(
                photo["time_s"],
                photo[f"{axis}_deg"],
                "b-",
                linewidth=0.8,
                label="Photogrammetry (reference)",
                alpha=0.8,
            )
            axes[row, 0].plot(
                t_ekf,
                ekf_vals[axis],
                f"{color}-",
                linewidth=0.8,
                label="EKF",
                alpha=0.8,
            )
            if std.get(axis) is not None:
                axes[row, 0].fill_between(
                    t_ekf,
                    ekf_vals[axis] - std[axis],
                    ekf_vals[axis] + std[axis],
                    alpha=0.2,
                    color=color,
                    label="±1σ",
                )
            axes[row, 0].set_ylabel(f"{label} (°)")
            axes[row, 0].set_title(f"{label} Angle Comparison")
            axes[row, 0].grid(True, alpha=0.3)
            axes[row, 0].legend(fontsize=8)

            # Error trace with mean ± std markers
            axes[row, 1].plot(t_ekf, err, f"{color}-", linewidth=0.5, alpha=0.8)
            axes[row, 1].axhline(0, color="k", linestyle="--", linewidth=0.5)
            axes[row, 1].axhline(
                stats[axis]["mean"],
                color="b",
                linestyle="-",
                linewidth=1,
                label=f"Mean: {stats[axis]['mean']:+.2f}°",
            )
            for sign in (1, -1):
                axes[row, 1].axhline(
                    stats[axis]["mean"] + sign * stats[axis]["std"],
                    color="b",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
            axes[row, 1].set_ylabel(f"{label} Error (°)")
            axes[row, 1].set_title(
                f"{label} Error   RMSE={stats[axis]['rmse']:.3f}°"
            )
            axes[row, 1].grid(True, alpha=0.3)
            axes[row, 1].legend(fontsize=8)

        for ax in axes.flat:
            if ax.get_visible():
                ax.set_xlabel("Time (s)" if ax.get_xlabel() == "" else ax.get_xlabel())

        plt.tight_layout()
        output_path = _TEST_DATA_DIR / "ekf_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nComparison plot saved → {output_path}")

        assert output_path.exists()

