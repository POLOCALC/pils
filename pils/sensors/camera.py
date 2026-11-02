import logging
import os
import glob
from pathlib import Path
from queue import Queue, Full
from threading import Thread
from types import SimpleNamespace
import multiprocessing as mp
from datetime import datetime, timezone

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

        # Reader thread is started lazily on first get_next_frame() call --
        # opening the capture + telemetry here must NOT start decoding, or
        # every flight prepared upfront (see Camera.build_multi_flight-style
        # callers) would buffer up to `buffer_size` 4K frames in RAM at once.

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

        # self._read_alvium_tstart()

        self.alvium_log = self._parse_alvium_log(self.logpath)
        # camera_data = read_alvium_log_time(keyphrase="captured.", logfile=self.logpath)
        camera_data = read_alvium_log_time(self.logpath)

        if camera_data.height > 0:
            # naive UTC, like the tstart from _read_alvium_tstart
            self.tstart = datetime.fromtimestamp(
                camera_data["timestamp"].min(), tz=timezone.utc
            ).replace(tzinfo=None)
            logger.info(f"Alvium tstart: {self.tstart}")
        else:
            logger.warning(f"No host_time_ns frame lines found in {self.logpath}")
            self.tstart = None

        self.fps = self._estimate_fps_from_log(camera_data)

        return camera_data, "alvium"

    @staticmethod
    def _estimate_fps_from_log(camera_data: pl.DataFrame) -> float:
        """Derive the true capture rate from consecutive frame timestamps.

        Falls back to 1.0 (the nominal rover_config default) if there
        aren't enough timestamps to measure an interval from -- the
        acquisition config (e.g. rover_config_2fps_halfroi.yml) can
        override the real rate, so it must never be hardcoded here.
        """
        if camera_data.height < 2:
            return 1.0

        dt = camera_data["timestamp"].diff().median()
        if dt is None or dt <= 0:
            return 1.0

        return 1.0 / dt






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
                self._enqueue(None)
                break
            if not self._enqueue((frame_idx, frame)):
                break  # release() was called while we were waiting to enqueue
            frame_idx += 1

    def _enqueue(self, item) -> bool:
        """Put ``item`` on ``frame_queue``, polling ``self.stopped`` so a full
        queue doesn't block the reader thread forever after ``release()``."""
        while not self.stopped:
            try:
                self.frame_queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False

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
            if self.stopped:
                return None  # released -- don't restart or block forever
            if self.reader_thread is None:
                self.reader_thread = Thread(target=self._reader_loop, daemon=True)
                self.reader_thread.start()
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
        if self.reader_thread is not None and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=5)
        if self.capture is not None:
            self.capture.release()

    # ------------------------------------------------------------------
    # Photogrammetry: moved to pils.analyze.photogrammetry
    # ------------------------------------------------------------------

    def run_photogrammetry(self, csv_file, config, flight, output_dir=None, **kwargs) -> pl.DataFrame:
        """Deprecated: use ``PhotogrammetryAnalysis(flight).run_analysis(...)``
        (pils.analyze.photogrammetry). Returns the merged result DataFrame, as before."""
        import warnings

        from pils.analyze.photogrammetry import PhotogrammetryAnalysis

        warnings.warn(
            "Camera.run_photogrammetry moved to "
            "pils.analyze.photogrammetry.PhotogrammetryAnalysis(flight).run_analysis",
            DeprecationWarning,
            stacklevel=2,
        )
        version = PhotogrammetryAnalysis(flight).run_analysis(
            csv_file=csv_file, config=config, output_dir=output_dir, **kwargs
        )
        return version.photogrammetry_data

    def __repr__(self) -> str:
        mode  = "image_sequence" if self.is_image_sequence else "video"
        model = self.data[1] if self.data is not None else "not loaded"
        return f"Camera(path={self.path}, mode={mode}, model={model})"


def __getattr__(name):
    # PhotogrammetryConfig moved to pils.analyze.photogrammetry; keep
    # `from pils.sensors.camera import PhotogrammetryConfig` working.
    if name == "PhotogrammetryConfig":
        from pils.analyze.photogrammetry import PhotogrammetryConfig

        return PhotogrammetryConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
