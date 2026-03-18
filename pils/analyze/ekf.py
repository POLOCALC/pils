"""EKF (Extended Kalman Filter) analysis module for flight attitude estimation."""

import datetime
import io
import json
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import polars as pl
import pyarrow as pa

from pils.flight import Flight
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)

# Path to the compiled Rust EKF binary (relative to this module)
_EKF_BIN_DIR = Path(__file__).parent / "ekf" / "bin"
_EKF_BINARY_PATH = _EKF_BIN_DIR / "rust-ekf"
_EKF_DEFAULT_CONFIG = _EKF_BIN_DIR / "config.yaml"

# Arrow IPC schemas for IMU and photogrammetry data
IMU_SCHEMA = pa.schema(
    [
        ("monotonic_ns", pa.int64()),
        ("timestamp_ns", pa.int64()),
        ("pqr_P_rad_s", pa.float64()),
        ("pqr_Q_rad_s", pa.float64()),
        ("pqr_R_rad_s", pa.float64()),
        ("acc_X_m_s2", pa.float64()),
        ("acc_Y_m_s2", pa.float64()),
        ("acc_Z_m_s2", pa.float64()),
    ]
)

PHOTO_SCHEMA = pa.schema(
    [
        ("monotonic_ns", pa.int64()),
        ("timestamp_ns", pa.int64()),
        ("quat_w", pa.float64()),
        ("quat_x", pa.float64()),
        ("quat_y", pa.float64()),
        ("quat_z", pa.float64()),
    ]
)

EKF_OUTPUT_SCHEMA = pa.schema(
    [
        ("timestamp_s", pa.float64()),
        ("timestamp_monotonic_ns", pa.float64()),
        ("roll_deg", pa.float64()),
        ("pitch_deg", pa.float64()),
        ("yaw_deg", pa.float64()),
        ("euler_cov_0_0", pa.float64()),
        ("euler_cov_0_1", pa.float64()),
        ("euler_cov_0_2", pa.float64()),
        ("euler_cov_1_0", pa.float64()),
        ("euler_cov_1_1", pa.float64()),
        ("euler_cov_1_2", pa.float64()),
        ("euler_cov_2_0", pa.float64()),
        ("euler_cov_2_1", pa.float64()),
        ("euler_cov_2_2", pa.float64()),
    ]
)


@dataclass
class EKFVersion:
    """EKF analysis version container.

    Parameters
    ----------
    version_name : str
        Name identifier for this EKF analysis version.
    ekf_data : pl.DataFrame
        Polars DataFrame with EKF output columns (timestamps, quaternions,
        Euler angles, gyro biases, etc.).
    metadata : dict[str, Any]
        Dictionary containing EKF configuration parameters and run metadata.

    Examples
    --------
    >>> ekf_data = pl.DataFrame({
    ...     'timestamp': [1000.0, 2000.0],
    ...     'quat_w': [1.0, 0.99],
    ...     'quat_x': [0.0, 0.01],
    ...     'quat_y': [0.0, 0.01],
    ...     'quat_z': [0.0, 0.01],
    ... })
    >>> metadata = {'config_file': 'config.yaml', 'num_imu_samples': 5000}
    >>> version = EKFVersion('v1', ekf_data, metadata)
    """

    version_name: str
    ekf_data: pl.DataFrame
    metadata: dict[str, Any]


class EKFAnalysis:
    """EKF (Extended Kalman Filter) attitude estimation analysis.

    Manages EKF-based attitude estimation by sending IMU and
    photogrammetry data to a compiled Rust EKF binary via Apache
    Arrow IPC. Completely separate from Flight class.

    File Structure
    --------------
    flight_dir/proc/ekf/
    └── (EKF analysis outputs will be stored here)

    Attributes
    ----------
    flight : Flight
        Flight object containing flight data.
    flight_path : Path
        Root flight directory.
    ekf_dir : Path
        EKF directory ({flight_path}/proc/ekf/).

    Examples
    --------
    >>> from pils.flight import Flight
    >>> # Create Flight object
    >>> flight_info = {
    ...     "drone_data_folder_path": "/path/to/flight/drone",
    ... }
    >>> flight = Flight(flight_info)
    >>> # Create new analysis
    >>> ekf = EKFAnalysis(flight)
    """

    def __init__(self, flight: Flight) -> None:
        """Initialize EKF analysis with Flight object.

        Creates the proc/ekf directory structure if it doesn't exist.

        Parameters
        ----------
        flight : Flight
            Flight object with valid flight_path attribute.

        Raises
        ------
        TypeError
            If flight is not a Flight object.
        ValueError
            If flight_path is None or doesn't exist.

        Examples
        --------
        >>> from pils.flight import Flight
        >>> flight_info = {
        ...     "drone_data_folder_path": "/path/to/flight/drone",
        ... }
        >>> flight = Flight(flight_info)
        >>> ekf = EKFAnalysis(flight)
        >>> print(ekf.ekf_dir)
        /path/to/flight/proc/ekf
        """
        # Validate flight object type
        if not isinstance(flight, Flight):
            raise TypeError(
                f"Expected Flight object, got {type(flight).__name__}. "
                "EKFAnalysis requires a Flight object instead of a path."
            )

        # Validate flight_path attribute exists
        if flight.flight_path is None:
            raise ValueError(
                "Flight object must have a valid flight_path attribute. "
                "Ensure the Flight was initialized with proper flight_info."
            )

        # Convert to Path and validate it's a directory
        flight_path = Path(flight.flight_path)
        if not flight_path.exists() or not flight_path.is_dir():
            raise ValueError(
                f"flight_path must be an existing directory. Got: {flight_path}"
            )

        # Store attributes
        self.flight = flight
        self.flight_path = flight_path

        # Create EKF directory
        self.ekf_dir = self.flight_path / "proc" / "ekf"
        self.ekf_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized EKF analysis for flight: {self.flight_path}")
        logger.info(f"EKF directory: {self.ekf_dir}")

    @staticmethod
    def _table_to_ipc_bytes(table: pa.Table, schema: pa.Schema) -> bytes:
        """Serialize an Arrow Table to IPC file-format bytes.

        Parameters
        ----------
        table : pa.Table
            PyArrow Table to serialize.
        schema : pa.Schema
            Arrow schema to use for the IPC file.

        Returns
        -------
        bytes
            IPC file-format bytes representing the table.

        Examples
        --------
        >>> import pyarrow as pa
        >>> schema = pa.schema([('x', pa.float64())])
        >>> table = pa.table({'x': [1.0, 2.0]})
        >>> ipc_bytes = EKFAnalysis._table_to_ipc_bytes(table, schema)
        """
        buf = io.BytesIO()
        with pa.ipc.new_file(buf, schema) as writer:
            writer.write_table(table)
        return buf.getvalue()

    @staticmethod
    def _prepare_imu_table(imu_df: pl.DataFrame) -> pa.Table:
        """Prepare IMU DataFrame as an Arrow Table with the required schema.

        Expects the IMU DataFrame to contain gyroscope and accelerometer
        columns with monotonic and timestamp nanosecond columns.

        Parameters
        ----------
        imu_df : pl.DataFrame
            Polars DataFrame with IMU data. Expected columns:
            monotonic_ns, timestamp_ns, pqr_P_rad_s, pqr_Q_rad_s,
            pqr_R_rad_s, acc_X_m_s2, acc_Y_m_s2, acc_Z_m_s2.

        Returns
        -------
        pa.Table
            Arrow Table conforming to IMU_SCHEMA.

        Raises
        ------
        ValueError
            If required columns are missing from the DataFrame.

        Examples
        --------
        >>> imu_df = pl.DataFrame({
        ...     'monotonic_ns': [0, 10000000],
        ...     'timestamp_ns': [1000000000, 1010000000],
        ...     'pqr_P_rad_s': [0.01, 0.02],
        ...     'pqr_Q_rad_s': [0.01, 0.02],
        ...     'pqr_R_rad_s': [0.01, 0.02],
        ...     'acc_X_m_s2': [0.0, 0.1],
        ...     'acc_Y_m_s2': [0.0, 0.1],
        ...     'acc_Z_m_s2': [-9.81, -9.80],
        ... })
        >>> table = EKFAnalysis._prepare_imu_table(imu_df)
        """
        required_cols = [
            "monotonic_ns",
            "timestamp_ns",
            "pqr_P_rad_s",
            "pqr_Q_rad_s",
            "pqr_R_rad_s",
            "acc_X_m_s2",
            "acc_Y_m_s2",
            "acc_Z_m_s2",
        ]
        missing = [col for col in required_cols if col not in imu_df.columns]
        if missing:
            raise ValueError(f"IMU DataFrame missing required columns: {missing}")

        # Select, order, and cast columns directly via Polars → Arrow (no pandas)
        return imu_df.select(required_cols).to_arrow().cast(IMU_SCHEMA)

    @staticmethod
    def _prepare_photo_table(photo_df: pl.DataFrame) -> pa.Table:
        """Prepare photogrammetry DataFrame as an Arrow Table.

        Expects the photogrammetry DataFrame to contain quaternion
        orientation columns with monotonic and timestamp nanosecond columns.

        Parameters
        ----------
        photo_df : pl.DataFrame
            Polars DataFrame with photogrammetry data. Expected columns:
            monotonic_ns, timestamp_ns, quat_w, quat_x, quat_y, quat_z.

        Returns
        -------
        pa.Table
            Arrow Table conforming to PHOTO_SCHEMA.

        Raises
        ------
        ValueError
            If required columns are missing from the DataFrame.

        Examples
        --------
        >>> photo_df = pl.DataFrame({
        ...     'monotonic_ns': [0, 50000000],
        ...     'timestamp_ns': [1000000000, 1050000000],
        ...     'quat_w': [1.0, 0.99],
        ...     'quat_x': [0.0, 0.01],
        ...     'quat_y': [0.0, 0.01],
        ...     'quat_z': [0.0, 0.01],
        ... })
        >>> table = EKFAnalysis._prepare_photo_table(photo_df)
        """
        required_cols = [
            "monotonic_ns",
            "timestamp_ns",
            "quat_w",
            "quat_x",
            "quat_y",
            "quat_z",
        ]
        missing = [col for col in required_cols if col not in photo_df.columns]
        if missing:
            raise ValueError(
                f"Photogrammetry DataFrame missing required columns: {missing}"
            )

        # Select, order, and cast columns directly via Polars → Arrow (no pandas)
        return photo_df.select(required_cols).to_arrow().cast(PHOTO_SCHEMA)

    @staticmethod
    def _send_ipc_to_rust(
        imu_bytes: bytes,
        photo_bytes: bytes,
        output_dir: Path,
        config_file: Path,
    ) -> subprocess.CompletedProcess:
        """Launch Rust EKF binary and send IMU + photogrammetry data via IPC stdin.

        Uses a length-prefixed protocol:
        [8-byte LE uint64: imu_len][imu IPC bytes]
        [8-byte LE uint64: photo_len][photo IPC bytes]

        Parameters
        ----------
        imu_bytes : bytes
            Serialized IMU Arrow IPC bytes.
        photo_bytes : bytes
            Serialized photogrammetry Arrow IPC bytes.
        output_dir : Path
            Directory where Rust EKF will write output files.
        config_file : Path
            Path to the EKF YAML configuration file.

        Returns
        -------
        subprocess.CompletedProcess
            Completed process with stdout, stderr, and returncode.

        Raises
        ------
        FileNotFoundError
            If the Rust EKF binary is not found.
        RuntimeError
            If the Rust process exits with a non-zero return code.

        Examples
        --------
        >>> result = EKFAnalysis._send_ipc_to_rust(
        ...     imu_bytes, photo_bytes,
        ...     output_dir=Path('/tmp/ekf_out'),
        ...     config_file=Path('config.yaml'),
        ... )
        """
        # Validate binary exists
        if not _EKF_BINARY_PATH.exists():
            raise FileNotFoundError(
                f"Rust EKF binary not found at {_EKF_BINARY_PATH}. "
                "Ensure the binary is compiled (cargo build --release)."
            )

        # Launch Rust process — omit --log to keep stdout clean for Arrow IPC.
        # Pass --log manually when running the binary directly for debugging.
        logger.info("Launching Rust EKF binary with IPC input...")
        process = subprocess.Popen(
            [
                str(_EKF_BINARY_PATH),
                "--ipc-input",
                "--output-dir",
                str(output_dir),
                "--config-file",
                str(config_file),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Send length-prefixed IPC payloads
        try:
            # IMU: 8-byte LE length + IPC payload
            process.stdin.write(struct.pack("<Q", len(imu_bytes)))
            process.stdin.write(imu_bytes)

            # Photogrammetry: 8-byte LE length + IPC payload
            process.stdin.write(struct.pack("<Q", len(photo_bytes)))
            process.stdin.write(photo_bytes)

            process.stdin.flush()
            logger.info("IMU + photogrammetry data sent to Rust EKF via IPC")

        except BrokenPipeError as err:
            raise RuntimeError(
                "Rust EKF process closed the connection unexpectedly. "
                "Check binary compatibility and input data."
            ) from err

        # communicate() flushes, closes stdin, reads stdout/stderr, and waits
        stdout, stderr = process.communicate()

        # Log process output — stdout carries raw Arrow IPC binary, not text
        if stderr:
            logger.info(f"Rust EKF stderr:\n{stderr.decode()}")
        if stdout:
            logger.info(f"Rust EKF stdout: {len(stdout)} bytes of Arrow IPC data")

        # Check return code
        if process.returncode != 0:
            raise RuntimeError(
                f"Rust EKF exited with code {process.returncode}. "
                f"stderr: {stderr.decode()}"
            )

        return subprocess.CompletedProcess(
            args=process.args,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    @staticmethod
    def _load_ekf_output(ipc_bytes: bytes) -> pl.DataFrame:
        """Load EKF output from Arrow IPC bytes returned on stdout.

        Reads the length-prefixed Arrow IPC payload written by the Rust
        binary to stdout and converts it to a Polars DataFrame.

        Wire format (little-endian):
        [8-byte LE uint64: payload_len][payload_len bytes: IPC file]

        Parameters
        ----------
        ipc_bytes : bytes
            Raw bytes from the Rust binary stdout, containing a
            length-prefixed Arrow IPC payload.

        Returns
        -------
        pl.DataFrame
            DataFrame with EKF output columns matching EKF_OUTPUT_SCHEMA:
            timestamp_s, timestamp_monotonic_ns, roll_deg, pitch_deg,
            yaw_deg, euler_cov_0_0 through euler_cov_2_2.

        Raises
        ------
        ValueError
            If no Arrow IPC payload can be located in ipc_bytes.

        Examples
        --------
        >>> result = EKFAnalysis._send_ipc_to_rust(...)
        >>> ekf_data = EKFAnalysis._load_ekf_output(result.stdout)
        >>> print(ekf_data.columns)
        ['timestamp_s', 'timestamp_monotonic_ns', 'roll_deg', ...]
        """
        _ARROW_MAGIC = b"ARROW1\x00\x00"

        if len(ipc_bytes) < 8:
            raise ValueError(
                f"EKF stdout too short for length header: {len(ipc_bytes)} bytes. "
                "Expected at least 8 bytes for the length prefix."
            )

        # ── Try length-prefix protocol (primary path) ─────────────────────────
        # Wire format: [8-byte LE uint64: payload_len][Arrow IPC file bytes]
        payload_len = struct.unpack("<Q", ipc_bytes[:8])[0]
        if payload_len > 0 and len(ipc_bytes) >= 8 + payload_len:
            payload = ipc_bytes[8 : 8 + payload_len]
            if payload[:8] == _ARROW_MAGIC:
                reader = pa.ipc.open_file(io.BytesIO(payload))
                return pl.from_arrow(reader.read_all())

        # ── Fallback: scan for Arrow magic bytes ──────────────────────────────
        # Handles any stdout prepended before the IPC payload (e.g. stray
        # println! or polars progress output in the Rust binary).
        idx = ipc_bytes.find(_ARROW_MAGIC)
        if idx < 0:
            raise ValueError(
                f"Arrow IPC magic bytes not found in EKF stdout "
                f"({len(ipc_bytes):,} bytes received). "
                "Ensure the Rust binary is up to date (cargo build --release)."
            )
        if idx > 0:
            logger.warning(
                f"EKF stdout: {idx} bytes of non-IPC data before Arrow payload "
                "(stdout pollution detected — check for println! in the Rust binary)"
            )

        reader = pa.ipc.open_file(io.BytesIO(ipc_bytes[idx:]))
        return pl.from_arrow(reader.read_all())

    def run_analysis(
        self,
        imu_df: pl.DataFrame,
        photo_df: pl.DataFrame,
        config_file: Path | str | None = None,
        save_data: bool = False,
    ) -> EKFVersion | None:
        """Run EKF attitude estimation analysis.

        Sends IMU and photogrammetry data to the Rust EKF binary
        via Apache Arrow IPC, then collects and stores the filtered
        attitude output.

        Parameters
        ----------
        imu_df : pl.DataFrame
            IMU data with columns: monotonic_ns, timestamp_ns,
            pqr_P_rad_s, pqr_Q_rad_s, pqr_R_rad_s,
            acc_X_m_s2, acc_Y_m_s2, acc_Z_m_s2.
        photo_df : pl.DataFrame
            Photogrammetry data with columns: monotonic_ns,
            timestamp_ns, quat_w, quat_x, quat_y, quat_z.
        config_file : Path | str | None, optional
            Path to EKF YAML configuration file.
            Default is None (uses built-in config.yaml).
        save_data : bool, optional
            Whether to save the data and create new version.
            Default is False.

        Returns
        -------
        EKFVersion | None
            EKFVersion with filtered attitude data and metadata,
            or None if processing fails.

        Raises
        ------
        ValueError
            If input DataFrames are empty or missing required columns.
        FileNotFoundError
            If Rust EKF binary or config file not found.
        RuntimeError
            If Rust EKF process fails.

        Examples
        --------
        >>> from pils.flight import Flight
        >>> from pils.analyze.ekf import EKFAnalysis
        >>> flight = Flight(flight_info)
        >>> ekf = EKFAnalysis(flight)
        >>> version = ekf.run_analysis(imu_df, photo_df)
        >>> print(version.ekf_data.head())
        >>> # Shows: timestamp, quaternion, Euler angle columns
        """

        logger.info("Starting EKF analysis...")

        # 1. Validate input DataFrames
        if imu_df is None or (isinstance(imu_df, pl.DataFrame) and imu_df.height == 0):
            raise ValueError("IMU DataFrame is empty or None")

        if photo_df is None or (
            isinstance(photo_df, pl.DataFrame) and photo_df.height == 0
        ):
            raise ValueError("Photogrammetry DataFrame is empty or None")

        logger.info(f"IMU data: {imu_df.height} rows, columns: {imu_df.columns}")
        logger.info(
            f"Photogrammetry data: {photo_df.height} rows, columns: {photo_df.columns}"
        )

        # 2. Resolve config file path
        if config_file is None:
            config_file = _EKF_DEFAULT_CONFIG
        else:
            config_file = Path(config_file)

        if not config_file.exists():
            raise FileNotFoundError(
                f"EKF config file not found: {config_file}. "
                "Provide a valid config YAML or use the default."
            )

        logger.info(f"Using EKF config: {config_file}")

        # 3. Prepare Arrow IPC tables from input DataFrames
        imu_table = self._prepare_imu_table(imu_df)
        photo_table = self._prepare_photo_table(photo_df)

        logger.info(f"Prepared IMU Arrow table: {imu_table.num_rows} rows")
        logger.info(f"Prepared photogrammetry Arrow table: {photo_table.num_rows} rows")

        # 4. Serialize tables to IPC bytes
        imu_bytes = self._table_to_ipc_bytes(imu_table, IMU_SCHEMA)
        photo_bytes = self._table_to_ipc_bytes(photo_table, PHOTO_SCHEMA)

        logger.info(f"Serialized IMU IPC: {len(imu_bytes)} bytes")
        logger.info(f"Serialized photogrammetry IPC: {len(photo_bytes)} bytes")

        # 5. Create output directory for this run
        output_dir = self.ekf_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 6. Send data to Rust EKF binary via IPC
        result = self._send_ipc_to_rust(imu_bytes, photo_bytes, output_dir, config_file)

        logger.info(f"Rust EKF completed with return code: {result.returncode}")

        # 7. Load EKF output data from stdout Arrow IPC
        ekf_data = self._load_ekf_output(result.stdout)

        logger.info(f"Loaded EKF output: {ekf_data.height} rows")

        # 8. Create metadata
        metadata = {
            "config_file": str(config_file),
            "num_imu_samples": imu_df.height,
            "num_photo_samples": photo_df.height,
            "num_output_samples": ekf_data.height,
            "imu_columns": imu_df.columns,
            "photo_columns": photo_df.columns,
            "output_columns": ekf_data.columns,
        }

        # 9. Generate version name (timestamp-based: rev_YYYYMMDD_HHMMSS)
        version_name = datetime.datetime.now().strftime("rev_%Y%m%d_%H%M%S")

        logger.info(
            f"EKF analysis complete: {ekf_data.height} output samples, "
            f"version={version_name}"
        )

        # Create version object
        version = EKFVersion(
            version_name=version_name, ekf_data=ekf_data, metadata=metadata
        )

        # Save to HDF5 if requested
        if save_data:
            self._save_to_hdf5(version)

        return version

    def _save_to_hdf5(self, version: EKFVersion) -> None:
        """Save EKF version to HDF5 file.

        Parameters
        ----------
        version : EKFVersion
            EKFVersion object to save to HDF5 file.

        Examples
        --------
        >>> version = ekf.run_analysis(...)
        >>> ekf._save_to_hdf5(version)
        """
        hdf5_path = self.ekf_dir / "ekf_solution.h5"

        with h5py.File(hdf5_path, "a") as f:
            # Create version group (overwrite if exists)
            if version.version_name in f:
                del f[version.version_name]

            version_group = f.create_group(version.version_name)

            # Save DataFrame columns
            df_group = version_group.create_group("ekf_data")
            for col_name in version.ekf_data.columns:
                col_data = version.ekf_data[col_name].to_numpy()
                df_group.create_dataset(col_name, data=col_data)

            # Save DataFrame schema info
            df_group.attrs["columns"] = json.dumps(version.ekf_data.columns)
            df_group.attrs["dtypes"] = json.dumps(
                [str(dtype) for dtype in version.ekf_data.dtypes]
            )

            # Save metadata
            version_group.attrs["metadata"] = json.dumps(version.metadata)

        logger.info(f"Saved EKF version '{version.version_name}' to HDF5")

    def _load_from_hdf5(self, version_name: str) -> EKFVersion:
        """Load EKF version from HDF5 file.

        Parameters
        ----------
        version_name : str
            Name of version to load (e.g., 'rev_20260218_143022').

        Returns
        -------
        EKFVersion
            Loaded EKFVersion object containing ekf_data and metadata.

        Raises
        ------
        FileNotFoundError
            If HDF5 file doesn't exist.
        KeyError
            If version_name not found in HDF5 file.

        Examples
        --------
        >>> version = ekf._load_from_hdf5('rev_20260218_143022')
        """
        hdf5_path = self.ekf_dir / "ekf_solution.h5"

        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            if version_name not in f:
                raise KeyError(f"Version '{version_name}' not found in HDF5")

            version_group = f[version_name]
            df_group = version_group["ekf_data"]

            # Reconstruct DataFrame from columns
            columns = json.loads(df_group.attrs["columns"])
            dtypes = json.loads(df_group.attrs["dtypes"])

            data_dict = {}
            for col_name in columns:
                data_dict[col_name] = df_group[col_name][:]

            ekf_data = pl.DataFrame(data_dict)

            # Cast to original dtypes
            cast_exprs = []
            for col_name, dtype_str in zip(columns, dtypes, strict=True):
                if "Float64" in dtype_str:
                    cast_exprs.append(pl.col(col_name).cast(pl.Float64))
                elif "Int64" in dtype_str:
                    cast_exprs.append(pl.col(col_name).cast(pl.Int64))

            if cast_exprs:
                ekf_data = ekf_data.with_columns(cast_exprs)

            # Load metadata
            metadata = json.loads(version_group.attrs["metadata"])

        logger.info(f"Loaded EKF version '{version_name}' from HDF5")

        return EKFVersion(
            version_name=version_name, ekf_data=ekf_data, metadata=metadata
        )

    def list_versions(self) -> list[str]:
        """List all EKF versions in HDF5 file.

        Returns
        -------
        list[str]
            List of version names, sorted chronologically.
            Empty list if HDF5 file doesn't exist.

        Examples
        --------
        >>> ekf.list_versions()
        ['rev_20260218_120000', 'rev_20260218_143022']
        """
        hdf5_path = self.ekf_dir / "ekf_solution.h5"

        if not hdf5_path.exists():
            return []

        with h5py.File(hdf5_path, "r") as f:
            versions = sorted(list(f.keys()))

        return versions

    def get_latest_version(self) -> EKFVersion | None:
        """Get the most recent EKF version.

        Returns
        -------
        EKFVersion | None
            Latest EKFVersion object, or None if no versions exist.

        Examples
        --------
        >>> latest = ekf.get_latest_version()
        >>> if latest:
        ...     print(latest.version_name)
        """
        versions = self.list_versions()

        if not versions:
            return None

        latest_name = versions[-1]  # Sorted chronologically, last is latest
        return self._load_from_hdf5(latest_name)
