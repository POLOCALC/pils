import logging
import os
import glob
from pathlib import Path
from queue import Queue
from threading import Thread
from types import SimpleNamespace
import multiprocessing as mp

import cv2
import numpy as np
import polars as pl
import pandas as pd

from ahrs import Quaternion
from ahrs.common.orientation import acc2q
from ahrs.filters import Madgwick
import telemetry_parser

from pils.utils.tools import read_alvium_log_time, read_log_time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PhotogrammetryConfig
# ---------------------------------------------------------------------------

class PhotogrammetryConfig:
    """Load and expose photogrammetry pipeline parameters from a YAML config file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file containing camera intrinsics
        and IPA_flight pipeline parameters.
    camera_model : str | None
        Explicit camera model key (e.g. ``"sony"`` or ``"alvium"``).
        If ``None``, the single camera block present in the YAML is used.

    Attributes
    ----------
    camera_matrix : np.ndarray
        3×3 camera intrinsic matrix.
    distortion_coeffs : np.ndarray
        Distortion coefficients array.
    raw : dict
        Full parsed YAML content.
    """

    def __init__(self, config_path: str | Path, camera_model: str | None = None) -> None:
        import yaml

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self.camera_model = camera_model

        with open(self.config_path, "r") as f:
            self.raw = yaml.safe_load(f)

        self._parse()

    # def _parse(self) -> None:
    #     model_key = self._resolve_camera_model()
    #     cam = self.raw.get(model_key, {})
    #     if not cam:
    #         raise ValueError(
    #             f"Camera calibration block '{model_key}' not found in {self.config_path}"
    #         )
    #     self.camera_matrix = np.array(cam["camera_matrix"], dtype=np.float64)
    #     self.distortion_coeffs = np.array(cam["distortion_coeffs"], dtype=np.float64)

    #     pipeline = self.raw.get("pipeline", {})
    #     self.reference_point = np.array(pipeline["reference_point"], dtype="double")

    #     self.finder = pipeline.get("finder", {})
    #     self.pnp = pipeline.get("pnp", {})
    #     self.mcmc = pipeline.get("mcmc", {})
    #     self.drone_correlation = pipeline.get("drone_correlation", {})
    #     self.telescope = pipeline.get("telescope", {})
    #     self.polarization = pipeline.get("polarization", {})

    def _parse(self) -> None:
        model_key = self._resolve_camera_model()
        cam = self.raw.get(model_key, {})
        if not cam:
            raise ValueError(
                f"Camera calibration block '{model_key}' not found in {self.config_path}"
            )
        self.camera_matrix = np.array(cam["camera_matrix"], dtype=np.float64)
        self.distortion_coeffs = np.array(cam["distortion_coeffs"], dtype=np.float64)

        pipeline = self.raw.get("pipeline", {})
        self.reference_point = np.array(pipeline["reference_point"], dtype="double")

        # ← ADD THIS
        raw_base = pipeline.get("dji_base_logged")
        self.dji_base_logged = np.array(raw_base, dtype="double") if raw_base is not None else None

        self.finder = pipeline.get("finder", {})
        self.pnp = pipeline.get("pnp", {})
        self.mcmc = pipeline.get("mcmc", {})
        self.drone_correlation = pipeline.get("drone_correlation", {})
        self.telescope = pipeline.get("telescope", {})
        self.polarization = pipeline.get("polarization", {})

    def _resolve_camera_model(self) -> str:
        if self.camera_model is not None:
            model = self.camera_model.lower()
            return model if model.endswith("_camera") else f"{model}_camera"

        camera_keys = [k for k in self.raw.keys() if k.endswith("_camera")]
        if len(camera_keys) == 1:
            return camera_keys[0]

        raise ValueError(
            "Config contains multiple camera calibrations. "
            "Pass camera_model='sony' or 'alvium' to PhotogrammetryConfig."
        )

    def __repr__(self) -> str:
        return (
            f"PhotogrammetryConfig(config={self.config_path}, "
            f"camera_matrix={self.camera_matrix.shape})"
        )


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    """Camera sensor for video files and image sequences.

    Supports four operating modes:

    1. **Photogrammetry results mode** (``use_photogrammetry=True``):
       Loads pre-processed photogrammetry CSV.

    2. **Sony RX0 MarkII mode** (``.mp4`` / ``.MP4`` files found):
       Extracts IMU telemetry, computes orientation via AHRS Madgwick filter.
       Starts a background reader thread for streaming.
       Falls back to empty DataFrame if telemetry is unavailable.

    3. **Alvium industrial camera mode** (image sequence + ``.log`` file):
       Reads frame timestamps from log file.

    4. **Photogrammetry pipeline mode** (``run_photogrammetry``):
       Orchestrates the full IPA_flight pipeline using a ``PhotogrammetryConfig``
       and a CSV of geodetic targets.
    """

    def __init__(
        self,
        path: str | Path,
        use_photogrammetry: bool = False,
        buffer_size: int = 128,
    ) -> None:
        self.path = Path(path)
        self.use_photogrammetry = use_photogrammetry

        self.capture = None
        self.fps: float | None = None
        self.tstart = None
        self._image_cursor: int = 0

        self.is_image_sequence: bool = False
        self.images: list[str] = []
        self.alvium_log: dict | pl.DataFrame = {}

        self._buffer_size = buffer_size
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.reader_thread: Thread | None = None
        self.stopped: bool = False

        self.data: tuple[pl.DataFrame, str | None] | None = None
        self.logpath: Path | None = None

    # ------------------------------------------------------------------
    # load_data
    # ------------------------------------------------------------------

    # def load_data(self) -> None:
    #     """Detect camera type and load data into ``self.data``."""
    #     if not self.path.exists():
    #         raise FileNotFoundError(f"Camera path does not exist: {self.path}")

    #     if self.use_photogrammetry:
    #         camera_data, camera_model = self._load_photogrammetry_data()
    #     else:
    #         video_files = [p for p in self.path.iterdir() if p.suffix.lower() == ".mp4"]
    #         if video_files:
    #             camera_data, camera_model = self._load_sony_camera_data(video_files)
    #         else:
    #             camera_data, camera_model = self._load_alvium_camera_data()

    #     self.data = (camera_data, camera_model)

    def load_data(self) -> None:
        """Detect camera type and load data into ``self.data``."""
        if not self.path.exists():
            raise FileNotFoundError(f"Camera path does not exist: {self.path}")

        if self.use_photogrammetry:
            camera_data, camera_model = self._load_photogrammetry_data()
        else:
            video_files = [p for p in self.path.iterdir() if p.suffix.lower() == ".mp4"]

            if video_files:
                camera_data, camera_model = self._load_sony_camera_data(video_files)
            else:
                # Check for Alvium: log in self.path, images in ../../proc/images/
                log_files  = list(self.path.glob("*.[Ll][Oo][Gg]"))
                images_dir = self.path.parent.parent / "proc" / "images"
                image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
                image_files = (
                    [p for p in images_dir.iterdir() if p.suffix.lower() in image_extensions]
                    if images_dir.exists() else []
                )

                if log_files and image_files:
                    camera_data, camera_model = self._load_alvium_camera_data()
                elif log_files:
                    # Log present but no images — raise clearly
                    raise FileNotFoundError(
                        f"Alvium log found in {self.path} but no images in {images_dir}"
                    )
                else:
                    raise FileNotFoundError(
                        f"No recognised camera data found in {self.path}. "
                        "Expected .mp4 files (Sony) or Alvium log + proc/images/."
                    )

        self.data = (camera_data, camera_model)




    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------

    def _load_photogrammetry_data(self) -> tuple[pl.DataFrame, None]:
        if self.path.is_dir():
            csv_files = list(self.path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV found in {self.path}")
            csv_path = csv_files[0]
        else:
            csv_path = self.path
        logger.info(f"Loading photogrammetry data from {csv_path}")
        return pl.read_csv(csv_path), None

    # def _load_sony_camera_data(self, video_files: list[Path]) -> tuple[pl.DataFrame, str]:
    #     log_file = list(self.path.parent.glob("*.[Ll][Oo][Gg]"))
    #     if not log_file:
    #         raise FileNotFoundError(
    #             f"No log file found for Sony camera in {self.path.parent}"
    #         )

    #     self.logpath = log_file[0]
    #     time_start, _ = read_log_time(
    #         keyphrase="Camera Sony starts recording", logfile=self.logpath
    #     )

    #     video_path = str(video_files[0])
    #     self.capture = cv2.VideoCapture(video_path)
    #     if not self.capture.isOpened():
    #         raise IOError(f"Cannot open video {video_path}")

    #     fps = self.capture.get(cv2.CAP_PROP_FPS)
    #     self.fps = fps if fps > 0 else None
    #     self.tstart = time_start

    #     self.reader_thread = Thread(target=self._reader_loop, daemon=True)
    #     self.reader_thread.start()

    #     try:
    #         camera_data = self._parse_sony_telemetry(video_path)
    #     except Exception as e:
    #         logger.warning(
    #             f"Sony telemetry unavailable for {video_path} ({e}). "
    #             "Storing empty DataFrame — streaming still works."
    #         )
    #         camera_data = pl.DataFrame()

    #     if time_start is not None and len(camera_data) > 0:
    #         camera_data = camera_data.with_columns(
    #             (pl.col("timestamp_ms") / 1000.0 + time_start.timestamp()).alias("timestamp")
    #         )

    #     return camera_data, "sony"

    def _load_sony_camera_data(self, video_files: list[Path]) -> tuple[pl.DataFrame, str]:
        log_file = list(self.path.parent.glob("*.[Ll][Oo][Gg]"))
        if not log_file:
            raise FileNotFoundError(
                f"No log file found for Sony camera in {self.path.parent}"
            )

        self.logpath = log_file[0]
        time_start, _ = read_log_time(
            keyphrase="Camera Sony starts recording", logfile=self.logpath
        )

        video_path = str(video_files[0])
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise IOError(f"Cannot open video {video_path}")

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps > 0 else None
        self.tstart = time_start

        # ✅ Parse telemetry FIRST, before any threads are started
        try:
            camera_data = self._parse_sony_telemetry(video_path)
        except Exception as e:
            logger.warning(
                f"Sony telemetry unavailable for {video_path} ({e}). "
                "Storing empty DataFrame — streaming still works."
            )
            camera_data = pl.DataFrame()

        # ✅ Start reader thread only after subprocess work is done
        self.reader_thread = Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

        if time_start is not None and len(camera_data) > 0:
            camera_data = camera_data.with_columns(
                (pl.col("timestamp_ms") / 1000.0 + time_start.timestamp()).alias("timestamp")
            )

        return camera_data, "sony"

    # def _load_alvium_camera_data(self) -> tuple[pl.DataFrame, str]:
    #     log_file = list(self.path.glob("*.[Ll][Oo][Gg]"))
    #     if not log_file:
    #         raise FileNotFoundError(f"No video files or log files found in {self.path}")

    #     self.logpath = log_file[0]
    #     self.is_image_sequence = True
    #     self.images = sorted(glob.glob(os.path.join(str(self.path), "*.*")))

    #     if not self.images:
    #         raise FileNotFoundError(f"No images found in {self.path}")

    #     self.fps = 1.0
    #     self.alvium_log = self._parse_alvium_log(self.logpath)
    #     camera_data = read_alvium_log_time(keyphrase="Saving frame", logfile=self.logpath)
    #     return camera_data, "alvium"

    def _load_alvium_camera_data(self) -> tuple[pl.DataFrame, str]:
        log_file = list(self.path.glob("*.[Ll][Oo][Gg]"))
        if not log_file:
            raise FileNotFoundError(f"No log file found for Alvium camera in {self.path}")

        self.logpath = log_file[0]

        flight_root = self.path.parent.parent
        images_dir  = flight_root / "proc" / "images"

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Expected Alvium images directory not found: {images_dir}"
            )

        image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        self.images = sorted(
            str(p) for p in images_dir.iterdir()
            if p.suffix.lower() in image_extensions
        )

        if not self.images:
            raise FileNotFoundError(f"No images found in {images_dir}")

        self.is_image_sequence = True
        self.fps = 1.0

        self._read_alvium_tstart()

        self.alvium_log = self._parse_alvium_log(self.logpath)
        camera_data = read_alvium_log_time(keyphrase="captured.", logfile=self.logpath)
        return camera_data, "alvium"






    def _read_alvium_tstart(self) -> None:
        """Parse tstart from Alvium log format: [2025-12-11 13:54:16.011 - INFO]"""
        import re
        pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) - INFO\]")
        
        try:
            with open(self.logpath, "r") as f:
                for line in f:
                    if "Started image acquisition." in line:
                        match = pattern.search(line)
                        if match:
                            from datetime import datetime
                            self.tstart = datetime.strptime(
                                match.group(1), "%Y-%m-%d %H:%M:%S.%f"
                            )
                            logger.info(f"Alvium tstart: {self.tstart}")
                            return
            logger.warning("Keyphrase 'Started image acquisition.' not found in Alvium log.")
            self.tstart = None
        except Exception as e:
            logger.warning(f"Could not read Alvium tstart from log: {e}")
            self.tstart = None

    def _parse_alvium_log(self, logpath: Path) -> pl.DataFrame | None:
        if logpath is None or not Path(logpath).is_file():
            return None
        try:
            df = pd.read_csv(str(logpath), sep=None, engine="python")
            first_col = df.columns[0]
            try:
                df[first_col] = pd.to_datetime(
                    df[first_col], format="mixed", utc=True
                )
            except Exception:
                try:
                    df[first_col] = pd.to_datetime(
                        df[first_col], format="ISO8601", utc=True
                    )
                except Exception:
                    pass  # leave column as-is if all parsing fails
            return pl.from_pandas(df)
        except Exception as e:
            logger.warning(f"Failed parsing Alvium log: {e}")
            return None


    # def _parse_alvium_log(self, logpath: Path) -> pl.DataFrame | None:
    #     if logpath is None or not Path(logpath).is_file():
    #         return None
    #     try:
    #         df = pd.read_csv(str(logpath), sep=None, engine="python")
    #         first_col = df.columns[0]
    #         try:
    #             df[first_col] = pd.to_datetime(df[first_col], utc=True)
    #         except Exception:
    #             pass
    #         return pl.from_pandas(df)
    #     except Exception as e:
    #         logger.warning(f"Failed parsing Alvium log: {e}")
    #         return None

    # ------------------------------------------------------------------
    # Sony telemetry
    # ------------------------------------------------------------------

    # def _parse_sony_telemetry(self, path: str) -> pl.DataFrame:
    #     try:
    #         parser = telemetry_parser.Parser(path)  # type: ignore
    #         imu_data = parser.normalized_imu()
    #     except Exception as e:
    #         logger.error(f"Failed to parse Sony telemetry from {path}: {e}")
    #         raise

    #     rows = []
    #     for entry in imu_data:
    #         gyro = entry.get("gyro", [])
    #         accl = entry.get("accl", [])
    #         if len(gyro) != 3 or len(accl) != 3:
    #             logger.warning(f"Skipping malformed IMU entry: gyro={gyro}, accl={accl}")
    #             continue
    #         rows.append({
    #             "timestamp_ms": entry["timestamp_ms"],
    #             "gyro_x": gyro[0], "gyro_y": gyro[1], "gyro_z": gyro[2],
    #             "accel_x": accl[0], "accel_y": accl[1], "accel_z": accl[2],
    #         })

    #     if not rows:
    #         raise ValueError(f"No valid IMU entries found in {path}")

    #     df = pl.DataFrame(rows)

    #     gyro_data  = df.select(["gyro_x", "gyro_y", "gyro_z"]).to_numpy()
    #     accel_data = df.select(["accel_x", "accel_y", "accel_z"]).to_numpy()
    #     timestamps = df["timestamp_ms"].to_numpy() / 1000.0

    #     dt        = np.mean(np.diff(timestamps))
    #     frequency = 1.0 / dt

    #     madgwick    = Madgwick(frequency=frequency)
    #     num_samples = len(df)
    #     Q           = np.zeros((num_samples, 4))
    #     Q[0]        = acc2q(accel_data[0])

    #     for t in range(1, num_samples):
    #         Q[t] = madgwick.updateIMU(Q[t - 1], gyr=gyro_data[t], acc=accel_data[t])

    #     euler_angles = np.array([Quaternion(q).to_angles() for q in Q])

    #     df = df.with_columns([
    #         pl.Series("roll",  euler_angles[:, 0]),
    #         pl.Series("pitch", euler_angles[:, 1]),
    #         pl.Series("yaw",   euler_angles[:, 2]),
    #         pl.Series("qw", Q[:, 0]),
    #         pl.Series("qx", Q[:, 1]),
    #         pl.Series("qy", Q[:, 2]),
    #         pl.Series("qz", Q[:, 3]),
    #     ])

    #     return df

    def _parse_sony_telemetry(self, path: str) -> pl.DataFrame:
        """
        Parse Sony IMU telemetry safely.

        Uses subprocess isolation because telemetry_parser may trigger
        unrecoverable Rust panics through PyO3.

        Data is exchanged via a temporary .npy file instead of mp.Queue
        to avoid pickling 100k dicts across the process boundary.
        """
        import time
        import tempfile
        import os

        def _worker(video_path: str, result_path: str, queue):
            try:
                parser   = telemetry_parser.Parser(video_path)
                imu_data = parser.normalized_imu()

                rows = []
                for entry in imu_data:
                    gyro = entry.get("gyro", [])
                    accl = entry.get("accl", [])
                    if len(gyro) != 3 or len(accl) != 3:
                        continue
                    try:
                        rows.append([
                            float(entry["timestamp_ms"]),
                            float(gyro[0]), float(gyro[1]), float(gyro[2]),
                            float(accl[0]), float(accl[1]), float(accl[2]),
                        ])
                    except Exception:
                        continue

                if not rows:
                    queue.put(("err", "No valid IMU entries found"))
                    return

                arr = np.array(rows, dtype=np.float64)
                np.save(result_path, arr)
                queue.put(("ok", len(rows)))

            except Exception as e:
                queue.put(("err", str(e)))

        # ------------------------------------------------------------------
        # File size based timeout
        # ------------------------------------------------------------------
        file_size_mb = Path(path).stat().st_size / 1_048_576
        timeout_s    = max(60, file_size_mb * 0.5)
        logger.info(
            f"Parsing telemetry from {path} "
            f"({file_size_mb:.1f} MB, timeout={timeout_s:.0f}s)"
        )

        # ------------------------------------------------------------------
        # Run parser in isolated subprocess
        # ------------------------------------------------------------------
        tmp_path = tempfile.mktemp(suffix=".npy")
        queue    = mp.Queue()
        proc     = mp.Process(target=_worker, args=(path, tmp_path, queue))
        proc.start()

        # ------------------------------------------------------------------
        # Progress heartbeat while waiting
        # ------------------------------------------------------------------
        t_start = time.monotonic()
        try:
            while proc.is_alive():
                elapsed = time.monotonic() - t_start
                if elapsed > timeout_s:
                    proc.terminate()
                    proc.join()
                    raise RuntimeError(
                        f"telemetry-parser timed out after {timeout_s:.0f}s "
                        f"on {path}"
                    )
                print(
                    f"   ⏳ telemetry_parser running... "
                    f"{elapsed:.0f}s / {timeout_s:.0f}s",
                    flush=True,
                )
                proc.join(timeout=10)

            # ------------------------------------------------------------------
            # Check exit code
            # ------------------------------------------------------------------
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"telemetry-parser crashed (exitcode={proc.exitcode}) "
                    f"on {path}"
                )

            # ------------------------------------------------------------------
            # Drain queue — only a tiny status message now, not 100k dicts
            # ------------------------------------------------------------------
            try:
                status, payload = queue.get(timeout=10)
            except Exception:
                raise RuntimeError(
                    f"telemetry-parser finished but returned no status for {path}"
                )

            if status == "err":
                raise RuntimeError(
                    f"Failed to parse Sony telemetry from {path}: {payload}"
                )

            n_samples = payload
            logger.info(f"Telemetry parsed: {n_samples} IMU samples")

            # ------------------------------------------------------------------
            # Load numpy array from temp file
            # ------------------------------------------------------------------
            if not os.path.exists(tmp_path):
                raise RuntimeError(
                    f"telemetry-parser succeeded but temp file missing: {tmp_path}"
                )

            arr = np.load(tmp_path)

        finally:
            # Always clean up temp file, even if an exception was raised
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # ------------------------------------------------------------------
        # Build dataframe
        # columns: timestamp_ms, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z
        # ------------------------------------------------------------------
        df = pl.DataFrame({
            "timestamp_ms": arr[:, 0],
            "gyro_x":       arr[:, 1],
            "gyro_y":       arr[:, 2],
            "gyro_z":       arr[:, 3],
            "accel_x":      arr[:, 4],
            "accel_y":      arr[:, 5],
            "accel_z":      arr[:, 6],
        })

        gyro_data  = arr[:, 1:4]
        accel_data = arr[:, 4:7]
        timestamps = arr[:, 0] / 1000.0

        # ------------------------------------------------------------------
        # Validate timestamps
        # ------------------------------------------------------------------
        if len(timestamps) < 2:
            raise ValueError(f"Not enough IMU samples in {path}")

        dts = np.diff(timestamps)

        if np.any(~np.isfinite(dts)):
            raise ValueError(f"Invalid timestamp deltas in {path}")

        if np.any(dts <= 0):
            raise ValueError(f"Non-monotonic timestamps in {path}")

        dt        = float(np.mean(dts))
        frequency = 1.0 / dt

        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError(f"Invalid frequency={frequency:.2f} Hz in {path}")

        logger.info(f"IMU frequency: {frequency:.2f} Hz  ({len(timestamps)} samples)")

        # ------------------------------------------------------------------
        # Validate accelerometer initialisation
        # ------------------------------------------------------------------
        first_acc = accel_data[0]

        if not np.all(np.isfinite(first_acc)):
            raise ValueError(f"Invalid initial accelerometer sample in {path}")

        if np.linalg.norm(first_acc) < 1e-6:
            raise ValueError(f"Zero initial accelerometer vector in {path}")

        # ------------------------------------------------------------------
        # Orientation estimation (Madgwick)
        # ------------------------------------------------------------------
        madgwick    = Madgwick(frequency=frequency)
        num_samples = len(df)
        Q           = np.zeros((num_samples, 4), dtype=np.float64)
        Q[0]        = acc2q(first_acc)

        for t in range(1, num_samples):
            try:
                Q[t] = madgwick.updateIMU(
                    Q[t - 1],
                    gyr=gyro_data[t],
                    acc=accel_data[t],
                )
            except Exception as e:
                logger.warning(f"Madgwick update failed at sample {t}: {e}")
                Q[t] = Q[t - 1]

        # ------------------------------------------------------------------
        # Euler angles
        # ------------------------------------------------------------------
        euler_angles = np.array([Quaternion(q).to_angles() for q in Q])

        # ------------------------------------------------------------------
        # Final dataframe
        # ------------------------------------------------------------------
        df = df.with_columns([
            pl.Series("roll",  euler_angles[:, 0]),
            pl.Series("pitch", euler_angles[:, 1]),
            pl.Series("yaw",   euler_angles[:, 2]),
            pl.Series("qw",    Q[:, 0]),
            pl.Series("qx",    Q[:, 1]),
            pl.Series("qy",    Q[:, 2]),
            pl.Series("qz",    Q[:, 3]),
        ])

        return df

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        frame_idx = 0
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                self.frame_queue.put(None)
                break
            self.frame_queue.put((frame_idx, frame))
            frame_idx += 1

    def get_next_frame(self) -> tuple[int, np.ndarray] | None:
        """Return the next frame as ``(frame_index, BGR array)``, or ``None`` when exhausted."""
        if self.is_image_sequence:
            if self._image_cursor >= len(self.images):
                return None
            frame = cv2.imread(self.images[self._image_cursor])
            idx = self._image_cursor
            self._image_cursor += 1
            return idx, frame
        else:
            return self.frame_queue.get()

    # ------------------------------------------------------------------
    # Random access (image sequence only)
    # ------------------------------------------------------------------

    def get_frame(self, frame_number: int) -> np.ndarray:
        if not self.is_image_sequence:
            raise RuntimeError("Random access is disabled for videos (streaming mode).")
        if frame_number < 0 or frame_number >= len(self.images):
            raise IndexError(f"Frame index {frame_number} out of range.")
        return cv2.imread(self.images[frame_number])

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def get_timestamp(self, frame_number: int):
        if not self.is_image_sequence:
            if self.tstart is None or self.fps is None:
                return None
            return self.tstart + pd.Timedelta(seconds=frame_number / self.fps)
        return None

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_frame(
        self,
        frame: np.ndarray | None = None,
        frame_number: int | None = None,
        color: str = "rgb",
        save_path: str | None = None,
    ) -> None:
        import matplotlib.pyplot as plt

        if frame is None:
            if frame_number is None:
                raise ValueError("Either frame or frame_number must be provided.")
            frame = self.get_frame(frame_number)

        converters = {
            "rgb":  cv2.COLOR_BGR2RGB,
            "hsv":  cv2.COLOR_BGR2HSV,
            "gray": cv2.COLOR_BGR2GRAY,
        }
        if color in converters:
            img = cv2.cvtColor(frame, converters[color])
        elif color == "bgr":
            img = frame.copy()
        else:
            raise KeyError(f"Unknown color space: {color}")

        plt.figure()
        plt.imshow(img, cmap="gray" if color == "gray" else None)
        ts = self.get_timestamp(frame_number) if frame_number is not None else "unknown"
        plt.title(f"Frame {frame_number if frame_number is not None else 'streamed'} — {ts}")
        plt.axis("off")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        self.stopped = True
        if self.capture is not None:
            self.capture.release()

    # ------------------------------------------------------------------
    # Internal: DataHandler wrapper builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dh_wrapper(flight, camera_obj: "Camera") -> SimpleNamespace:
        """Build the SimpleNamespace DataHandler wrapper that IPA_flight expects.

        Normalises drone and litchi DataFrames (polars → pandas, tz-naive
        datetimes) and wraps everything into a lightweight namespace that
        exposes ``dh.camera``, ``dh.raw_data``, and ``dh.flight_info``.
        """
        raw_data_attr = getattr(flight, "raw_data", None)

        if raw_data_attr is not None and hasattr(raw_data_attr, "drone_data"):
            drone_data_attr = raw_data_attr.drone_data

            # ── drone DataFrame ───────────────────────────────────────
            drone_df = getattr(drone_data_attr, "drone", None)
            try:
                if isinstance(drone_df, pl.DataFrame):
                    drone_df = drone_df.to_pandas()
            except Exception:
                pass

            # Normalise datetime to tz-naive pandas Timestamp
            try:
                if isinstance(drone_df, pd.DataFrame):
                    if "datetime" in drone_df.columns:
                        drone_df["datetime"] = pd.to_datetime(
                            drone_df["datetime"], errors="coerce", utc=True
                        )
                    elif "timestamp" in drone_df.columns:
                        drone_df["datetime"] = pd.to_datetime(
                            drone_df["timestamp"], unit="s", errors="coerce", utc=True
                        )
                    if "datetime" in drone_df.columns:
                        try:
                            drone_df["datetime"] = drone_df["datetime"].dt.tz_convert(None)
                        except Exception:
                            drone_df["datetime"] = drone_df["datetime"].dt.tz_localize(None)
            except Exception:
                pass

            # ── litchi DataFrame ──────────────────────────────────────
            litchi_raw = getattr(drone_data_attr, "litchi", None)
            litchi_df  = None

            if litchi_raw is not None:
                try:
                    if isinstance(litchi_raw, (pl.DataFrame, pl.Series)):
                        litchi_df = litchi_raw.to_pandas()
                except Exception:
                    pass

                if litchi_df is None:
                    if isinstance(litchi_raw, pd.DataFrame):
                        litchi_df = litchi_raw
                    elif isinstance(litchi_raw, pd.Series):
                        litchi_df = litchi_raw.to_frame()
                    else:
                        try:
                            litchi_df = pd.DataFrame(litchi_raw)
                        except Exception:
                            litchi_df = pd.DataFrame()

                try:
                    if isinstance(litchi_df, pd.DataFrame) and "datetime" in litchi_df.columns:
                        litchi_df["datetime"] = pd.to_datetime(
                            litchi_df["datetime"], errors="coerce", utc=True
                        )
                        try:
                            litchi_df["datetime"] = litchi_df["datetime"].dt.tz_convert(None)
                        except Exception:
                            litchi_df["datetime"] = litchi_df["datetime"].dt.tz_localize(None)
                except Exception:
                    pass

            litchi_obj = SimpleNamespace(data=litchi_df)
            drone_ns   = SimpleNamespace(drone=drone_df, litchi=litchi_obj)
            raw_ns     = SimpleNamespace(
                drone_data=drone_ns,
                payload_data=getattr(raw_data_attr, "payload_data", None),
            )
        else:
            raw_ns = raw_data_attr

        return SimpleNamespace(
            camera=camera_obj,
            raw_data=raw_ns,
            flight_info=getattr(flight, "flight_info", None),
        )

    # ------------------------------------------------------------------
    # Multi-flight entry point
    # ------------------------------------------------------------------

    @staticmethod
    def run_photogrammetry_multi_flights(
        flights: list,
        csv_file: str | Path,
        config: "PhotogrammetryConfig",
        output_dir: str | Path,
        start_from_dict: str | Path | None = None,
        target_indices: list[int] | None = None,
        mcmc_solution: bool = False,
    ) -> pl.DataFrame:
        """Run the full pipeline across multiple flights.

        Step p2 (TargetFinder) is executed **once** across all videos using
        ``finderSequentialMultiVideo``, so the user only needs to click targets
        on the first video; subsequent videos are aligned automatically via ORB.

        Both ``tg_by_hand="click_with_tel"`` and ``tg_by_hand="click"`` are
        handled: either way a reference frame is stored after the first video
        so that ORB alignment works for all subsequent ones.

        Steps p3–p7 run independently per flight so outputs land in separate
        per-flight subfolders under ``output_dir``.

        Parameters
        ----------
        flights : list of Flight
            All flights for the campaign, in chronological order.
        csv_file : str | Path
            CSV with geodetic targets and telescope positions.
        config : PhotogrammetryConfig
            Loaded config carrying camera intrinsics and pipeline parameters.
        output_dir : str | Path
            Root output directory; per-flight subfolders are created automatically.
        start_from_dict : str | Path | None
            Path to an intermediate ``.ecsv`` to resume from mid-pipeline.
            When set, step p2 is skipped entirely (targets loaded from file).

        Returns
        -------
        pl.DataFrame
            Concatenated final results across all flights with an extra
            ``flight_idx`` column.
        """
        from IPA_flight.IPA_flight.finder import TargetFinder
        from IPA_flight.IPA_flight.genParamFile import GenParamFile

        output_dir = Path(output_dir)

        # ── 1. Prepare all flights: drone + camera data ────────────────────
        wrappers      = []   # dh_wrapper per valid flight
        valid_flights = []   # (original_index, flight, camera_obj)

        for i, flight in enumerate(flights):
            print(f"\n{'='*60}\n  Preparing flight {i + 1}/{len(flights)}\n{'='*60}")

            try:
                flight.add_drone_data()
            except Exception as e:
                print(f"[SKIP] Flight {i + 1}: drone data failed: {e}")
                continue

            try:
                flight.add_camera_data(use_photogrammetry=False)
            except Exception as e:
                print(f"[SKIP] Flight {i + 1}: camera data failed: {e}")
                continue

            camera_obj = flight.raw_data.payload_data.camera_obj
            wrappers.append(Camera._build_dh_wrapper(flight, camera_obj))
            valid_flights.append((i, flight, camera_obj))

        if not wrappers:
            print("No valid flights could be prepared — aborting.")
            return pl.DataFrame()

        # ── 2. Build TargetFinder from first valid flight's params ─────────
        first_flight = valid_flights[0][1]
        # params = GenParamFile.GeoPlot(
        #     csv_file=str(csv_file),
        #     reference_point=config.reference_point,
        #     flight=first_flight,
        #     dji_base_logged=config.dji_base_logged
        # )

        # AFTER
        params = GenParamFile.GeoPlot(
                csv_file=str(csv_file),
                reference_point=config.reference_point,
                flight=first_flight,                   # ← fix 1: use first_flight
                dji_base_logged=getattr(config, "dji_base_logged", None),
            )

        geodetic_targets_filtered = (
            [params["geodetic_targets"][i] for i in target_indices]
            if target_indices is not None
            else params["geodetic_targets"]
        )

        target_finder = TargetFinder(
            wrappers[0],
            geodetic_targets_filtered,                 # ← fix 2: filtered list
            params["geodetic_tel_positions"],
            params["image_tel_positions"],
            config.camera_matrix,
            config.distortion_coeffs,
            params["reference_point"],
            params["video_code"],
        )

        # ── 3. Run multi-video tracking (step p2) ─────────────────────────
        # Strip tg_by_hand from finder kwargs — finderSequentialMultiVideo
        # takes it as a dedicated argument.
        finder_kwargs = {k: v for k, v in config.finder.items() if k != "tg_by_hand"}
        tg_by_hand    = config.finder.get("tg_by_hand", "click_with_tel")

        if start_from_dict:
            # Resume path: skip tracking, load targets from file
            from IPA_flight.IPA_flight.utils import load_dictionary
            print(f"\n⏭  Resuming from {start_from_dict} — skipping step p2.")
            all_targets_df = None   # signal to per-flight steps to load from file
        else:
            print(f"\n🎬 Running finderSequentialMultiVideo across {len(wrappers)} video(s) "
                  f"[mode: {tg_by_hand}] …")
            all_targets_df = target_finder.finderSequentialMultiVideo(
                video_list=wrappers,
                tg_by_hand=tg_by_hand,
                **finder_kwargs,
            )

        # ── 4. Run remaining pipeline steps per flight ─────────────────────
        all_results = []

        for list_idx, (orig_idx, flight, camera_obj) in enumerate(valid_flights):
            print(f"\n{'='*60}\n  Pipeline steps p3–p7: flight {orig_idx + 1}\n{'='*60}")

            # Slice this flight's tracking rows out of the combined DataFrame
            if all_targets_df is not None:
                flight_targets = all_targets_df[
                    all_targets_df["video_idx"] == list_idx
                ].drop(columns=["video_idx"])
            else:
                flight_targets = None  # run_photogrammetry will use start_from_dict

            try:
                result = camera_obj.run_photogrammetry(
                    csv_file=csv_file,
                    config=config,
                    flight=flight,
                    output_dir=output_dir,
                    check_results=None,
                    start_from_dict=start_from_dict,
                    precomputed_targets=flight_targets,
                    mcmc_solution=mcmc_solution,
                    target_indices=target_indices,
                )
                result = result.with_columns([pl.lit(orig_idx).alias("flight_idx")])
                all_results.append(result)
            except Exception as e:
                print(f"[SKIP] Flight {orig_idx + 1} pipeline steps failed: {e}")
                continue

        if not all_results:
            return pl.DataFrame()

        return pl.concat(all_results, how="vertical")

    # ------------------------------------------------------------------
    # Single-flight photogrammetry pipeline
    # ------------------------------------------------------------------

    def run_photogrammetry(
        self,
        csv_file: str | Path,
        config: "PhotogrammetryConfig",
        flight: "Flight",
        output_dir: str | Path = "/data/POLOCALC/processed_data/photogrammetry_results",
        gps_offset: list[float] | None = None,
        check_results: str | Path | None = None,
        start_from_dict: str | Path | None = None,
        precomputed_targets=None,
        target_indices: list[int] | None = None,
        mcmc_solution: bool = False,
    ) -> pl.DataFrame:
        """Run the full IPA_flight photogrammetry pipeline for a single flight.

        Parameters
        ----------
        csv_file : str | Path
            Path to CSV with geodetic targets and telescope positions.
        config : PhotogrammetryConfig
            Loaded config carrying camera intrinsics and pipeline parameters.
        flight : Flight
            The pils Flight object.
        output_dir : str | Path
            Root output directory; a per-flight subfolder is created automatically.
        gps_offset : list[float] | None
            Optional RTK base offset ``[dE, dN, dU]`` in ENU metres.
        check_results : str | Path | None
            If set, diagnostic plots are saved here.
            Defaults to ``<output_dir>/<flight_name>/plots/``.
        start_from_dict : str | Path | None
            Path to an intermediate ``.ecsv`` to resume from mid-pipeline.
        precomputed_targets : pd.DataFrame | pl.DataFrame | None
            When supplied by ``run_photogrammetry_multi_flights``, step p2
            (TargetFinder) is skipped and this DataFrame is used directly.

        Returns
        -------
        pl.DataFrame
            Final merged photogrammetry result.
        """
        from IPA_flight.IPA_flight.utils import (
            draw_results_on_last_frame,
            save_intermediate_results,
            load_dictionary,
            plot_attitude,
            plot_drone_gps_alignment,
            plot_telescope_attitude,
            plot_targets_coordinates,
            plot_telescope_pointing,
        )
        from IPA_flight.IPA_flight.finder import TargetFinder
        from IPA_flight.IPA_flight.attitude import AttitudeReconstruction
        from IPA_flight.IPA_flight.drone import DroneData as IPADroneData
        from IPA_flight.IPA_flight.polAng import PolarizationAngle
        from IPA_flight.IPA_flight.telescope import ConcerningTelescope
        from IPA_flight.IPA_flight.genParamFile import GenParamFile

        # ── Per-flight output subfolder ────────────────────────────────────
        flight_name = (
            flight.metadata.get("flight_name")
            or Path(flight.flight_info["drone_data_folder_path"]).parent.name
        )
        output_dir = Path(output_dir) / flight_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Photogrammetry output dir: {output_dir}")

        if check_results is None:
            check_results = output_dir / "plots"

        # ── Parse targets from CSV ─────────────────────────────────────────
        params = GenParamFile.GeoPlot(
                csv_file=str(csv_file),
                reference_point=config.reference_point,
                flight=flight,
                dji_base_logged=getattr(config, "dji_base_logged", None),
            )
        if target_indices is not None:
            params["geodetic_targets"] = [
                params["geodetic_targets"][i]
                for i in target_indices
            ]

            # if "image_tel_positions" in params:
            #     params["image_tel_positions"] = [
            #         params["image_tel_positions"][i]
            #         for i in target_indices
            #     ]

        gps_offset = params["gps_offset"]

        # ── Build IPA_flight modules ───────────────────────────────────────
        dh_wrapper = Camera._build_dh_wrapper(flight, self)

        target_finder = TargetFinder(
            dh_wrapper,
            params["geodetic_targets"],
            params["geodetic_tel_positions"],
            params["image_tel_positions"],
            config.camera_matrix,
            config.distortion_coeffs,
            params["reference_point"],
            params["video_code"],
        )

        attitude_reconstruction = AttitudeReconstruction(
            config.camera_matrix,
            config.distortion_coeffs,
        )

        drone_data = IPADroneData(flight, params["reference_point"])

        polarization_angle = PolarizationAngle(
            params["geodetic_tel_positions"][0],
            params["reference_point"],
        )

        telescope = ConcerningTelescope(
            params["geodetic_tel_positions"],
            params["reference_point"],
            tel_names=config.telescope.get(
                "tel_names", ["SATp1", "SATp2", "SATp3", "CLASS1", "CLASS2"]
            ),
        )

        # ── Pipeline step definitions ──────────────────────────────────────
        # Step p2 uses precomputed_targets when supplied (multi-flight path),
        # otherwise falls through to finderSequential (single-flight path).
        def _step_p2(_prev):
            if precomputed_targets is not None:
                logger.info("Step p2: using precomputed targets from multi-flight run.")
                return self._ensure_polars(precomputed_targets)
            return target_finder.finderSequential(**config.finder)
        
        if mcmc_solution:
            print("\n⚠️  Using MCMC solution for step p3 (this may be very slow) …")
            p3_step = lambda prev: attitude_reconstruction.run_mcmc(
                prev,
                **config.mcmc,
                )
        else:
            p3_step = lambda prev: attitude_reconstruction.run_pnp(
                prev,
                **config.pnp,
            )

        steps = {
            "dictionary_p2.ecsv": _step_p2,
           
            "dictionary_p3.ecsv": p3_step,
            
            "dictionary_p4.ecsv": lambda prev: drone_data.correlate_drone_photo(
                prev, gps_offset=gps_offset, **config.drone_correlation
            ),
            "dictionary_p5.ecsv": lambda prev: attitude_reconstruction._correct_tvec_with_gps(
                dataDict=prev,
                targetDict=load_dictionary(output_dir / "dictionary_p2.ecsv"),
            ),
            "dictionary_p6.ecsv": lambda prev: telescope.drone_in_telescope_frame(prev),
            "dictionary_p7.ecsv": lambda prev: polarization_angle.drone_attitude_in_LOS_frame(prev),
        }

        step_order = list(steps.keys())

        # ── Determine start point ──────────────────────────────────────────
        if start_from_dict:
            # Resolve to per-flight output dir if only a filename was given
            sfd_path = Path(os.path.expanduser(str(start_from_dict)))

            if not sfd_path.exists():
                sfd_resolved = output_dir / sfd_path.name

                if not sfd_resolved.exists():
                    raise FileNotFoundError(
                        f"Dictionary file '{start_from_dict}' not found. "
                        f"Also tried: {sfd_resolved}"
                    )

                sfd_path = sfd_resolved

            start_file = sfd_path.name

            if start_file not in step_order:
                raise ValueError(
                    f"Unknown starting dictionary: {start_file}"
                )

            start_index = step_order.index(start_file) + 1
            current_dict = self._ensure_polars(
                load_dictionary(str(sfd_path))
            )

            logger.info(
                f"Resuming photogrammetry from: {sfd_path}"
            )

        else:
            start_index = 0
            current_dict = None

        # ── Run pipeline ───────────────────────────────────────────────────
        for i in range(start_index, len(step_order)):
            step_name = step_order[i]
            logger.info(f"Photogrammetry step: {step_name}")

            current_dict = self._ensure_polars(steps[step_name](current_dict))

            save_intermediate_results(
                current_dict,
                str(output_dir / step_name),
                params["video_code"],
                params["date"],
                params["video_name"],
            )

            if check_results:
                self._run_check_results(
                    current_dict,
                    step_name,
                    Path(check_results),
                    draw_results_on_last_frame,
                    plot_targets_coordinates,
                    plot_attitude,
                    plot_drone_gps_alignment,
                    plot_telescope_pointing,
                    plot_telescope_attitude,
                )

        # ── Merge final results ────────────────────────────────────────────
        dicts = [
            self._ensure_polars(
                load_dictionary(str(output_dir / f"dictionary_p{i}.ecsv"))
            )
            for i in range(2, 8)
        ]

        df_merged = (
            dicts[0]
            .join(dicts[1], on="frame", how="outer")
            .join(dicts[2], on="frame", how="outer")
            .join(dicts[3], on="frame", how="outer")
            .join(dicts[4].drop("time"), on="frame", how="outer")
            .join(dicts[5].drop("time"), on="frame", how="outer")
        )

        out_parquet = output_dir / "attitude_reconstruction.parquet"
        df_merged.write_parquet(str(out_parquet))
        logger.info(f"Photogrammetry result saved: {out_parquet}")

        return df_merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_polars(d) -> pl.DataFrame:
        if isinstance(d, pd.DataFrame):
            return pl.from_pandas(d)
        return d

    def _run_check_results(
        self,
        current_dict: pl.DataFrame,
        step_name: str,
        plot_dir: Path,
        draw_results_on_last_frame,
        plot_targets_coordinates,
        plot_attitude,
        plot_drone_gps_alignment,
        plot_telescope_pointing,
        plot_telescope_attitude,
    ) -> None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        cols = current_dict.columns

        if "x" in cols and "y" in cols:
            plot_df = current_dict.to_pandas() if hasattr(current_dict, "to_pandas") else current_dict
            out     = str(plot_dir / f"last_frame_{step_name}.jpg")
            if self.is_image_sequence:
                draw_results_on_last_frame(plot_df, image_sequence=self.images, output_path=out)
            else:
                draw_results_on_last_frame(plot_df, video_path=str(self.path), output_path=out)
            plot_targets_coordinates(
                plot_df,
                output_path=str(plot_dir / f"targets_coords_{step_name}.jpg"),
            )

        if all(c in cols for c in ["rvec_x", "rvec_y", "rvec_z", "tvec_x", "tvec_y", "tvec_z"]):
            plot_attitude(
                current_dict,
                rolling_window=4,
                output_path=str(plot_dir / f"attitude_{step_name}.jpg"),
            )

        if all(c in cols for c in ["drone_E", "drone_N", "drone_U",
                                    "tvec_E", "tvec_N", "tvec_U", "time"]):
            plot_drone_gps_alignment(
                current_dict,
                outpath=str(plot_dir / f"drone_gps_{step_name}.jpg"),
            )

        if all(c in cols for c in ["tel_name", "time", "az", "el", "yaw", "pitch", "roll"]):
            plot_telescope_pointing(
                current_dict,
                outpath=str(plot_dir / f"telescope_pointing_{step_name}.jpg"),
            )

        if all(c in cols for c in ["yaw_LOS", "pitch_LOS", "roll_LOS"]):
            plot_telescope_attitude(
                current_dict,
                outpath=str(plot_dir / f"telescope_attitude_{step_name}.jpg"),
            )

    def __repr__(self) -> str:
        mode  = "image_sequence" if self.is_image_sequence else "video"
        model = self.data[1] if self.data is not None else "not loaded"
        return f"Camera(path={self.path}, mode={mode}, model={model})"