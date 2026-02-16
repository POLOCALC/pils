import logging
from pathlib import Path

import numpy as np
import polars as pl
import telemetry_parser
from ahrs import Quaternion
from ahrs.common.orientation import acc2q
from ahrs.filters import Madgwick

from pils.utils.tools import read_alvium_log_time, read_log_time

logger = logging.getLogger(__name__)


class Camera:
    """Camera sensor for video files and image sequences.

    Supports three operating modes:
    1. Photogrammetry mode: Loads pre-processed photogrammetry results from CSV
    2. Sony RX0 MarkII mode: Extracts IMU telemetry from .mp4 files and computes orientation
    3. Alvium industrial camera mode: Reads timestamp and frame number from log files

    The camera data is stored as a tuple (DataFrame, model_string) in the `data` attribute
    after calling load_data().

    Attributes
    ----------
    path : Path
        Path to video file, image directory, or photogrammetry CSV.
    use_photogrammetry : bool
        Whether to use photogrammetry mode.
    data : tuple[pl.DataFrame, str | None]
        Camera data as (DataFrame, camera_model). Set by load_data().
        camera_model is "sony", "alvium", or None for photogrammetry.
    logpath : Path, optional
        Path to log file for timestamp extraction (set during load_data).

    Examples
    --------
    >>> # Load Sony camera with IMU telemetry
    >>> camera = Camera("/path/to/camera/folder")
    >>> camera.load_data()
    >>> df, model = camera.data
    >>> print(model)  # "sony"
    >>> print(df.columns)  # ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', ...]
    >>>
    >>> # Load photogrammetry results
    >>> camera = Camera("/path/to/results.csv", use_photogrammetry=True)
    >>> camera.load_data()
    >>> df, model = camera.data
    >>> print(model)  # None
    >>> print(df.columns)  # ['timestamp', 'pitch', 'roll', 'yaw']
    """

    def __init__(self, path: str | Path, use_photogrammetry: bool = False) -> None:
        """Initialize Camera sensor.

        Parameters
        ----------
        path : str | Path
            Path to camera data. For Sony/Alvium cameras, this is a directory
            containing .mp4 files or .log files. For photogrammetry mode,
            this is the path to a CSV file with processed results.
        use_photogrammetry : bool, default False
            If True, loads photogrammetry CSV. If False, auto-detects Sony
            or Alvium camera based on files in directory.

        Examples
        --------
        >>> camera = Camera("/path/to/camera/folder")
        >>> camera = Camera("/path/to/photogrammetry.csv", use_photogrammetry=True)
        """
        self.path = Path(path)
        self.use_photogrammetry = use_photogrammetry

    def load_data(self) -> None:
        """Load camera data and store in self.data attribute.

        Automatically detects camera type and loads appropriate data:

        **Photogrammetry mode** (use_photogrammetry=True):
            Loads CSV file with columns: timestamp, pitch, roll, yaw
            Sets self.data = (DataFrame, None)

        **Sony RX0 MarkII mode** (.mp4 files found):
            Extracts IMU telemetry from video, computes orientation using AHRS
            Requires .log file in parent directory with start timestamp
            Sets self.data = (DataFrame, "sony")
            DataFrame columns: timestamp, gyro_x/y/z, accel_x/y/z, roll, pitch, yaw, qw/qx/qy/qz

        **Alvium industrial camera mode** (no .mp4, .log file found):
            Reads log file for frame timestamps and numbers
            Sets self.data = (DataFrame, "alvium")
            DataFrame columns: timestamp, frame_num

        Raises
        ------
        FileNotFoundError
            If the camera data path does not exist or no valid files are found.

        Examples
        --------
        >>> camera = Camera("/path/to/sony/folder")
        >>> camera.load_data()
        >>> df, model = camera.data
        >>> print(model)  # "sony"
        >>> df.select(['timestamp', 'pitch', 'roll', 'yaw'])
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Camera data path does not exist: {self.path}")

        if self.use_photogrammetry:
            camera_data, camera_model = self._load_photogrammetry_data()
        else:
            video_files = list(self.path.glob("*.[Mm][Pp]4"))

            if len(video_files) > 0:
                camera_data, camera_model = self._load_sony_camera_data(video_files)
            else:
                camera_data, camera_model = self._load_alvium_camera_data()

        self.data = (camera_data, camera_model)

    def _load_photogrammetry_data(self) -> tuple[pl.DataFrame, None]:
        """Load pre-processed photogrammetry data from CSV.

        Expected CSV format:
            Must contain at minimum: timestamp, pitch columns
            May also include: roll, yaw, x, y, z (position)

        Returns
        -------
        tuple[pl.DataFrame, None]
            DataFrame with photogrammetry data and None for camera model.

        Raises
        ------
        FileNotFoundError
            If CSV file does not exist.

        Examples
        --------
        >>> camera = Camera("photogrammetry.csv", use_photogrammetry=True)
        >>> df, model = camera._load_photogrammetry_data()
        >>> print(model)  # None
        >>> 'timestamp' in df.columns and 'pitch' in df.columns  # True
        """
        # If path is a directory, find CSV file inside it
        if self.path.is_dir():
            csv_files = list(self.path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError
            csv_path = csv_files[0]  # Use first CSV file found
            logger.info(f"Loading photogrammetry data from {csv_path}")
        else:
            csv_path = self.path

        try:
            camera_data = pl.read_csv(csv_path)
            return camera_data, None
        except FileNotFoundError:
            logger.info(f"Photogrammetry CSV file not found: {csv_path}")
            raise

    def _load_sony_camera_data(
        self, video_files: list[Path]
    ) -> tuple[pl.DataFrame, str]:
        """Load Sony RX0 MarkII camera data from video telemetry.

        Extracts embedded IMU data (gyroscope and accelerometer) from .mp4 video,
        computes orientation quaternions and Euler angles using AHRS Madgwick filter,
        and aligns timestamps with log file start time.

        Expected log file format (in parent directory):
            2024/11/20 14:30:45.123000 [INFO] Camera Sony starts recording

        Parameters
        ----------
        video_files : list[Path]
            List of video file paths (.mp4). Uses first file if multiple found.

        Returns
        -------
        tuple[pl.DataFrame, str]
            DataFrame with IMU data and orientation, "sony" model string.
            DataFrame columns: timestamp, timestamp_ms, gyro_x/y/z, accel_x/y/z,
                              roll, pitch, yaw, qw, qx, qy, qz

        Raises
        ------
        FileNotFoundError
            If no log file found in parent directory.

        Examples
        --------
        >>> video_files = [Path("/data/camera/video001.mp4")]
        >>> df, model = camera._load_sony_camera_data(video_files)
        >>> print(model)  # "sony"
        >>> df.select(['roll', 'pitch', 'yaw'])  # Orientation angles
        """
        log_file = list(self.path.parent.glob("*.[Ll][Oo][Gg]"))

        if len(log_file) == 0:
            logger.warning(f"No log file found in {self.path.parent}")
            raise FileNotFoundError(
                f"No log file found for Sony camera in {self.path.parent}"
            )

        self.logpath = log_file[0]

        time_start, _ = read_log_time(
            keyphrase="Camera Sony starts recording", logfile=self.logpath
        )

        camera_data = self._parse_sony_telemetry(str(video_files[0]))

        if time_start is not None:
            camera_data = camera_data.with_columns(
                (pl.col("timestamp_ms") / 1000.0 + time_start.timestamp()).alias(
                    "timestamp"
                )
            )

        return camera_data, "sony"

    def _load_alvium_camera_data(self) -> tuple[pl.DataFrame, str]:
        """Load Alvium industrial camera data from log file.

        Reads shooting log file to extract frame capture timestamps and frame numbers.

        Expected log file format:
            [2024-11-20 14:30:15.123] INFO: Saving frame frame_0001.raw
            [2024-11-20 14:30:15.223] INFO: Saving frame frame_0002.raw

        Returns
        -------
        tuple[pl.DataFrame, str]
            DataFrame with timestamp and frame_num columns, "alvium" model string.

        Raises
        ------
        FileNotFoundError
            If no log file found in camera directory.

        Examples
        --------
        >>> camera = Camera("/path/to/alvium/data")
        >>> df, model = camera._load_alvium_camera_data()
        >>> print(model)  # "alvium"
        >>> df.select(['timestamp', 'frame_num'])
        """
        log_file = list(self.path.glob("*.[Ll][Oo][Gg]"))

        if len(log_file) == 0:
            logger.warning(f"No log file found in {self.path}")
            raise FileNotFoundError(f"No video files or log files found in {self.path}")

        self.logpath = log_file[0]

        camera_data = read_alvium_log_time(
            keyphrase="Saving frame", logfile=self.logpath
        )

        return camera_data, "alvium"

    def _parse_sony_telemetry(self, path: str) -> pl.DataFrame:
        """Extract and process IMU telemetry from Sony RX0 MarkII video file.

        Uses telemetry-parser library to extract embedded gyroscope and accelerometer
        data from .mp4 video file. Applies AHRS Madgwick filter to compute orientation
        quaternions and converts to Euler angles (roll, pitch, yaw).

        The Madgwick filter is initialized with the sampling frequency calculated from
        timestamps, and the initial orientation is estimated from the first accelerometer
        reading.

        Parameters
        ----------
        path : str
            Path to Sony .mp4 video file with embedded IMU telemetry.

        Returns
        -------
        pl.DataFrame
            DataFrame with columns:
            - timestamp_ms: Relative timestamp in milliseconds from video start
            - gyro_x, gyro_y, gyro_z: Angular velocity (rad/s)
            - accel_x, accel_y, accel_z: Linear acceleration (m/s²)
            - roll, pitch, yaw: Euler angles (radians)
            - qw, qx, qy, qz: Orientation quaternion components

        Raises
        ------
        Exception
            If telemetry parsing fails or video file is corrupted.

        Examples
        --------
        >>> df = camera._parse_sony_telemetry("/path/to/video.mp4")
        >>> df.select(['timestamp_ms', 'roll', 'pitch', 'yaw'])
        >>> df['qw'].mean()  # Quaternion w component
        """
        try:
            parser = telemetry_parser.Parser(path)  # type: ignore
            imu_data = parser.normalized_imu()
        except Exception as e:
            logger.error(f"Failed to parse Sony telemetry from {path}: {e}")
            raise

        df = pl.DataFrame(
            [
                {
                    "timestamp_ms": entry["timestamp_ms"],
                    "gyro_x": entry["gyro"][0],
                    "gyro_y": entry["gyro"][1],
                    "gyro_z": entry["gyro"][2],
                    "accel_x": entry["accl"][0],
                    "accel_y": entry["accl"][1],
                    "accel_z": entry["accl"][2],
                }
                for entry in imu_data
            ]
        )

        # Extract gyro and accel data as numpy arrays
        gyro_data = df.select(
            ["gyro_x", "gyro_y", "gyro_z"]
        ).to_numpy()  # Shape: (n, 3)
        accel_data = df.select(
            ["accel_x", "accel_y", "accel_z"]
        ).to_numpy()  # Shape: (n, 3)

        # Get timestamps (convert ms to seconds if needed for frequency calculation)
        timestamps = df["timestamp_ms"].to_numpy() / 1000.0  # Convert to seconds

        # Calculate sampling frequency
        dt = np.mean(np.diff(timestamps))  # Average time step
        frequency = 1.0 / dt  # Hz

        # Initialize AHRS filter (e.g., Madgwick)
        madgwick = Madgwick(frequency=frequency)

        # Initialize quaternion array
        num_samples = len(df)
        Q = np.zeros((num_samples, 4))
        Q[0] = acc2q(accel_data[0])  # Initial orientation from accelerometer

        # Update quaternions
        for t in range(1, num_samples):
            Q[t] = madgwick.updateIMU(Q[t - 1], gyr=gyro_data[t], acc=accel_data[t])

        # Convert quaternions to Euler angles (roll, pitch, yaw)
        euler_angles = np.array([Quaternion(q).to_angles() for q in Q])  # Shape: (n, 3)

        # Add back to DataFrame
        df = df.with_columns(
            [
                pl.Series("roll", euler_angles[:, 0]),
                pl.Series("pitch", euler_angles[:, 1]),
                pl.Series("yaw", euler_angles[:, 2]),
                pl.Series("qw", Q[:, 0]),
                pl.Series("qx", Q[:, 1]),
                pl.Series("qy", Q[:, 2]),
                pl.Series("qz", Q[:, 3]),
            ]
        )

        return df
