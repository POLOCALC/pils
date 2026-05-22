import logging
import os
import glob
from pathlib import Path
from queue import Queue
from threading import Thread

import cv2
import numpy as np
import polars as pl
import pandas as pd

from ahrs import Quaternion
from ahrs.common.orientation import acc2q
from ahrs.filters import Madgwick
import telemetry_parser

from pils.utils.tools import read_alvium_log_time, read_log_time
from types import SimpleNamespace

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

    Attributes
    ----------
    camera_matrix : np.ndarray
        3x3 camera intrinsic matrix.
    distortion_coeffs : np.ndarray
        Distortion coefficients array.
    raw : dict
        Full parsed YAML content.

    Examples
    --------
    >>> cfg = PhotogrammetryConfig("config.yaml")
    >>> cfg.camera_matrix
    array([[fx,  0, cx],
           [ 0, fy, cy],
           [ 0,  0,  1]])
    """

    def __init__(self, config_path: str | Path) -> None:
        import yaml

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self.raw = yaml.safe_load(f)

        self._parse()

    def _parse(self) -> None:
        """Parse YAML content into typed attributes."""
        cam = self.raw.get("camera", {})
        self.camera_matrix = np.array(cam["camera_matrix"], dtype=np.float64)
        self.distortion_coeffs = np.array(cam["distortion_coeffs"], dtype=np.float64)

        pipeline = self.raw.get("pipeline", {})
        self.reference_point = np.array(pipeline["reference_point"], dtype="double")

        # IPA_flight pipeline kwargs — unpacked into each step
        self.finder = pipeline.get("finder", {})
        self.pnp = pipeline.get("pnp", {})
        self.drone_correlation = pipeline.get("drone_correlation", {})
        self.telescope = pipeline.get("telescope", {})
        self.polarization = pipeline.get("polarization", {})

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
       Reads frame timestamps from log file. Supports random-access and streaming.

    4. **Photogrammetry pipeline mode** (``run_photogrammetry``):
       Orchestrates the full IPA_flight pipeline using a ``PhotogrammetryConfig``
       and a CSV of geodetic targets.

    Attributes
    ----------
    path : Path
        Path to camera data directory or video file.
    use_photogrammetry : bool
        Whether to load pre-processed photogrammetry results.
    data : tuple[pl.DataFrame, str | None]
        ``(DataFrame, camera_model)`` populated by ``load_data()``.
        ``camera_model`` is ``"sony"``, ``"alvium"``, or ``None``.
    is_image_sequence : bool
        True when operating on an image sequence (Alvium mode).
    images : list[str]
        Sorted list of image file paths (image sequence mode only).

    Examples
    --------
    >>> camera = Camera("/path/to/alvium/folder")
    >>> camera.load_data()
    >>> df, model = camera.data   # model == "alvium"

    >>> cfg = PhotogrammetryConfig("config.yaml")
    >>> result = camera.run_photogrammetry(
    ...     csv_file="/path/to/targets.csv",
    ...     config=cfg,
    ...     flight=flight,
    ... )
    """

    def __init__(
        self,
        path: str | Path,
        use_photogrammetry: bool = False,
        buffer_size: int = 128,
    ) -> None:
        self.path = Path(path)
        self.use_photogrammetry = use_photogrammetry

        # Video / streaming state
        self.capture = None
        self.fps: float | None = None
        self.tstart = None
        self._image_cursor: int = 0

        # Image sequence state
        self.is_image_sequence: bool = False
        self.images: list[str] = []
        self.alvium_log: dict | pl.DataFrame = {}

        # Streaming
        self._buffer_size = buffer_size
        self.frame_queue: Queue = Queue(maxsize=buffer_size)
        self.reader_thread: Thread | None = None
        self.stopped: bool = False

        # Populated by load_data()
        self.data: tuple[pl.DataFrame, str | None] | None = None
        self.logpath: Path | None = None

    # ------------------------------------------------------------------
    # load_data
    # ------------------------------------------------------------------

    def load_data(self) -> None:
        """Detect camera type and load data into ``self.data``.

        Sets ``self.data = (DataFrame, camera_model)`` where
        ``camera_model`` is ``"sony"``, ``"alvium"``, or ``None``.

        Raises
        ------
        FileNotFoundError
            If the camera path does not exist or no valid files are found.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Camera path does not exist: {self.path}")

        if self.use_photogrammetry:
            camera_data, camera_model = self._load_photogrammetry_data()
        else:
            # Explicit case-insensitive search for MP4 files
            video_files = [
                p for p in self.path.iterdir()
                if p.suffix.lower() == ".mp4"
            ]
            if video_files:
                camera_data, camera_model = self._load_sony_camera_data(video_files)
            else:
                camera_data, camera_model = self._load_alvium_camera_data()

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

    def _load_sony_camera_data(
        self, video_files: list[Path]
    ) -> tuple[pl.DataFrame, str]:
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

        # Start background reader thread
        self.reader_thread = Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

        # Guard: not all MP4s contain Sony telemetry — fall back gracefully
        try:
            camera_data = self._parse_sony_telemetry(video_path)
        except Exception as e:
            logger.warning(
                f"Sony telemetry unavailable for {video_path} ({e}). "
                "Storing empty DataFrame — streaming still works."
            )
            camera_data = pl.DataFrame()

        if time_start is not None and len(camera_data) > 0:
            camera_data = camera_data.with_columns(
                (pl.col("timestamp_ms") / 1000.0 + time_start.timestamp()).alias(
                    "timestamp"
                )
            )

        return camera_data, "sony"

    def _load_alvium_camera_data(self) -> tuple[pl.DataFrame, str]:
        log_file = list(self.path.glob("*.[Ll][Oo][Gg]"))
        if not log_file:
            raise FileNotFoundError(
                f"No video files or log files found in {self.path}"
            )

        self.logpath = log_file[0]
        self.is_image_sequence = True
        self.images = sorted(glob.glob(os.path.join(str(self.path), "*.*")))

        if not self.images:
            raise FileNotFoundError(f"No images found in {self.path}")

        self.fps = 1.0  # Alvium 1 Hz
        self.alvium_log = self._parse_alvium_log(self.logpath)

        camera_data = read_alvium_log_time(
            keyphrase="Saving frame", logfile=self.logpath
        )

        return camera_data, "alvium"

    def _parse_alvium_log(self, logpath: Path) -> pl.DataFrame | None:
        """Parse Alvium log file into a Polars DataFrame."""
        if logpath is None or not Path(logpath).is_file():
            return None
        try:
            df = pd.read_csv(str(logpath), sep=None, engine="python")
            first_col = df.columns[0]
            try:
                df[first_col] = pd.to_datetime(df[first_col], utc=True)
            except Exception:
                pass
            return pl.from_pandas(df)
        except Exception as e:
            logger.warning(f"Failed parsing Alvium log: {e}")
            return None

    # ------------------------------------------------------------------
    # Sony telemetry
    # ------------------------------------------------------------------

    def _parse_sony_telemetry(self, path: str) -> pl.DataFrame:
        """Extract IMU telemetry from Sony RX0 MarkII .mp4 and compute orientation."""
        try:
            parser = telemetry_parser.Parser(path)  # type: ignore
            imu_data = parser.normalized_imu()
        except Exception as e:
            logger.error(f"Failed to parse Sony telemetry from {path}: {e}")
            raise

        rows = []
        for entry in imu_data:
            gyro = entry.get("gyro", [])
            accl = entry.get("accl", [])
            if len(gyro) != 3 or len(accl) != 3:
                logger.warning(f"Skipping malformed IMU entry: gyro={gyro}, accl={accl}")
                continue
            rows.append({
                "timestamp_ms": entry["timestamp_ms"],
                "gyro_x": gyro[0], "gyro_y": gyro[1], "gyro_z": gyro[2],
                "accel_x": accl[0], "accel_y": accl[1], "accel_z": accl[2],
            })

        if not rows:
            raise ValueError(f"No valid IMU entries found in {path}")

        df = pl.DataFrame(rows)

        gyro_data  = df.select(["gyro_x", "gyro_y", "gyro_z"]).to_numpy()
        accel_data = df.select(["accel_x", "accel_y", "accel_z"]).to_numpy()
        timestamps = df["timestamp_ms"].to_numpy() / 1000.0

        dt        = np.mean(np.diff(timestamps))
        frequency = 1.0 / dt

        madgwick    = Madgwick(frequency=frequency)
        num_samples = len(df)
        Q           = np.zeros((num_samples, 4))
        Q[0]        = acc2q(accel_data[0])

        for t in range(1, num_samples):
            Q[t] = madgwick.updateIMU(Q[t - 1], gyr=gyro_data[t], acc=accel_data[t])

        euler_angles = np.array([Quaternion(q).to_angles() for q in Q])

        df = df.with_columns([
            pl.Series("roll",  euler_angles[:, 0]),
            pl.Series("pitch", euler_angles[:, 1]),
            pl.Series("yaw",   euler_angles[:, 2]),
            pl.Series("qw", Q[:, 0]),
            pl.Series("qx", Q[:, 1]),
            pl.Series("qy", Q[:, 2]),
            pl.Series("qz", Q[:, 3]),
        ])

        return df

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _reader_loop(self) -> None:
        """Background thread: decode frames and push to queue."""
        frame_idx = 0
        while not self.stopped:
            ret, frame = self.capture.read()
            if not ret:
                self.frame_queue.put(None)
                break
            self.frame_queue.put((frame_idx, frame))
            frame_idx += 1

    def get_next_frame(self) -> tuple[int, np.ndarray] | None:
        """Return the next frame as ``(frame_index, BGR array)``.

        Works for both video (streaming) and image sequence modes.
        Returns ``None`` when exhausted.
        """
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
        """Return a single frame by index (image sequence only).

        Raises
        ------
        RuntimeError
            If called on a video (streaming mode).
        IndexError
            If ``frame_number`` is out of range.
        """
        if not self.is_image_sequence:
            raise RuntimeError("Random access is disabled for videos (streaming mode).")
        if frame_number < 0 or frame_number >= len(self.images):
            raise IndexError(f"Frame index {frame_number} out of range.")
        return cv2.imread(self.images[frame_number])

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def get_timestamp(self, frame_number: int):
        """Return the timestamp for a given frame number."""
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
        """Display or save a single frame."""
        import matplotlib.pyplot as plt

        if frame is None:
            if frame_number is None:
                raise ValueError("Either frame or frame_number must be provided.")
            frame = self.get_frame(frame_number)

        converters = {
            "rgb": cv2.COLOR_BGR2RGB,
            "hsv": cv2.COLOR_BGR2HSV,
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
        """Release video capture and stop background reader thread."""
        self.stopped = True
        if self.capture is not None:
            self.capture.release()

    # ------------------------------------------------------------------
    # Photogrammetry pipeline
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
    ) -> pl.DataFrame:
        """Run the full IPA_flight photogrammetry pipeline.

        Results are written to ``output_dir / <flight_name> /``.
        The final merged DataFrame is saved as ``attitude_reconstruction.parquet``.

        Parameters
        ----------
        csv_file : str | Path
            Path to CSV file with geodetic targets and telescope positions.
        config : PhotogrammetryConfig
            Loaded config carrying camera intrinsics and pipeline parameters.
        flight : Flight
            The pils Flight object, provides drone data and flight name.
        output_dir : str | Path
            Root output directory. A per-flight subfolder is created automatically.
            Defaults to ``/data/POLOCALC/processed_data/photogrammetry_results``.
        gps_offset : list[float] | None
            Optional RTK base offset as ``[dE, dN, dU]`` in ENU metres.
        check_results : str | Path | None
            If set, diagnostic plots are saved here.
            Defaults to ``<output_dir>/<flight_name>/plots/``.
        start_from_dict : str | Path | None
            Path to an intermediate ``.ecsv`` to resume from mid-pipeline.

        Returns
        -------
        pl.DataFrame
            Final merged photogrammetry result.

        Examples
        --------
        >>> cfg = PhotogrammetryConfig("pils/config/photogrammetry_config.yaml")
        >>> result = flight.raw_data.payload_data.camera_obj.run_photogrammetry(
        ...     csv_file="targets.csv",
        ...     config=cfg,
        ...     flight=flight,
        ... )
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

        # ---- Per-flight output subfolder -----------------------------------
        flight_name = (
            flight.metadata.get("flight_name")
            or Path(flight.flight_info["drone_data_folder_path"]).parent.name
        )
        output_dir = Path(output_dir) / flight_name
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Photogrammetry output dir: {output_dir}")

        # Default plots dir inside the flight output folder
        if check_results is None:
            check_results = output_dir / "plots"

        # ---- Parse targets from CSV ----------------------------------------
        params = GenParamFile.GeoPlot(
            csv_file=str(csv_file),
            reference_point=config.reference_point,
            flight=flight,
        )
        # GPS offset is derived from the CSV (Dji_base - Dji_base_selfmeasured)
        gps_offset = params["gps_offset"]

        # ---- Build IPA_flight modules ---------------------------------------
        # TargetFinder expects a DataHandler-like object exposing a `.camera`
        # attribute. Wrap this Camera instance into a lightweight namespace
        # so that IPA_flight code which accesses `DH.camera.*` continues to work.
        # Include `raw_data` and `flight_info` so DroneData recognizes this
        # wrapper as a `pils` Flight-like object (is_pils=True path).
        # Build a Flight-like raw_data wrapper that matches legacy expectations.
        raw_data_attr = getattr(flight, "raw_data", None)
        if raw_data_attr is not None and hasattr(raw_data_attr, "drone_data"):
            # drone_data may contain polars or pandas objects — convert to pandas
            drone_data_attr = raw_data_attr.drone_data

            # Convert drone dataframe to pandas if needed
            drone_df = getattr(drone_data_attr, "drone", None)
            try:
                import polars as _pl
                if isinstance(drone_df, _pl.DataFrame):
                    drone_df = drone_df.to_pandas()
            except Exception:
                pass

            # Normalize drone datetime/timestamp columns to pandas datetime (naive)
            try:
                if isinstance(drone_df, pd.DataFrame):
                    if "datetime" in drone_df.columns:
                        drone_df["datetime"] = pd.to_datetime(drone_df["datetime"], errors="coerce", utc=True)
                        try:
                            if drone_df["datetime"].dt.tz is not None:
                                drone_df["datetime"] = drone_df["datetime"].dt.tz_convert(None)
                        except Exception:
                            try:
                                drone_df["datetime"] = drone_df["datetime"].dt.tz_localize(None)
                            except Exception:
                                pass
                    elif "timestamp" in drone_df.columns:
                        drone_df["datetime"] = pd.to_datetime(drone_df["timestamp"], unit="s", errors="coerce", utc=True)
                        try:
                            if drone_df["datetime"].dt.tz is not None:
                                drone_df["datetime"] = drone_df["datetime"].dt.tz_convert(None)
                        except Exception:
                            try:
                                drone_df["datetime"] = drone_df["datetime"].dt.tz_localize(None)
                            except Exception:
                                pass
            except Exception:
                pass

            # Convert litchi to pandas DataFrame and ensure .data attribute exists
            litchi_raw = getattr(drone_data_attr, "litchi", None)
            litchi_df = None
            if litchi_raw is not None:
                try:
                    import polars as _pl
                    if isinstance(litchi_raw, _pl.DataFrame):
                        litchi_df = litchi_raw.to_pandas()
                    elif isinstance(litchi_raw, _pl.Series):
                        litchi_df = litchi_raw.to_pandas()
                except Exception:
                    pass

                if litchi_df is None:
                    if isinstance(litchi_raw, pd.DataFrame):
                        litchi_df = litchi_raw
                    elif isinstance(litchi_raw, pd.Series):
                        litchi_df = litchi_raw.to_frame()
                    else:
                        # Fallback: wrap raw into a single-column DataFrame
                        try:
                            litchi_df = pd.DataFrame(litchi_raw)
                        except Exception:
                            litchi_df = pd.DataFrame()
                # Normalize datetime columns to pandas.Timestamp (naive, no tz)
                try:
                    if isinstance(litchi_df, pd.DataFrame) and "datetime" in litchi_df.columns:
                        litchi_df["datetime"] = pd.to_datetime(litchi_df["datetime"], errors="coerce", utc=True)
                        try:
                            # convert tz-aware to naive
                            if litchi_df["datetime"].dt.tz is not None:
                                litchi_df["datetime"] = litchi_df["datetime"].dt.tz_convert(None)
                        except Exception:
                            try:
                                litchi_df["datetime"] = litchi_df["datetime"].dt.tz_localize(None)
                            except Exception:
                                pass
                except Exception:
                    pass

            litchi_obj = SimpleNamespace(data=litchi_df)
            drone_ns = SimpleNamespace(drone=drone_df, litchi=litchi_obj)
            raw_ns = SimpleNamespace(drone_data=drone_ns, payload_data=getattr(raw_data_attr, "payload_data", None))
        else:
            raw_ns = raw_data_attr

        dh_wrapper = SimpleNamespace(
            camera=self,
            raw_data=raw_ns,
            flight_info=getattr(flight, "flight_info", None),
        )

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

        # ---- Workflow steps -------------------------------------------------
        steps = {
            "dictionary_p2.ecsv": lambda prev: target_finder.finderSequential(
                tg_by_hand="click",
                **config.finder,
            ),
            "dictionary_p3.ecsv": lambda prev: attitude_reconstruction.run_pnp(
                prev,
                **config.pnp,
            ),
            "dictionary_p4.ecsv": lambda prev: drone_data.correlate_drone_photo(
                prev,
                gps_offset=gps_offset,
                **config.drone_correlation,
            ),
            "dictionary_p5.ecsv": lambda prev: attitude_reconstruction._correct_tvec_with_gps(
                dataDict=prev,
                targetDict=load_dictionary(output_dir / "dictionary_p2.ecsv"),
            ),
            "dictionary_p6.ecsv": lambda prev: telescope.drone_in_telescope_frame(prev),
            "dictionary_p7.ecsv": lambda prev: polarization_angle.drone_attitude_in_LOS_frame(prev),
        }

        step_order = list(steps.keys())

        # ---- Determine start point ------------------------------------------
        if start_from_dict:
            start_file = Path(start_from_dict).name
            if start_file not in step_order:
                raise ValueError(f"Unknown starting dictionary: {start_file}")
            start_index = step_order.index(start_file) + 1
            current_dict = self._ensure_polars(load_dictionary(start_from_dict))
            logger.info(f"Resuming photogrammetry from: {start_file}")
        else:
            start_index = 0
            current_dict = None
            logger.info("Starting photogrammetry pipeline from the beginning.")

        # ---- Run pipeline ---------------------------------------------------
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

        # ---- Merge final results --------------------------------------------
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

        # ---- Save ----------------------------------------------------------
        out_parquet = output_dir / "attitude_reconstruction.parquet"
        df_merged.write_parquet(str(out_parquet))
        logger.info(f"Photogrammetry result saved: {out_parquet}")

        return df_merged

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_polars(d) -> pl.DataFrame:
        """Convert pandas DataFrame to Polars if needed."""
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
        """Save diagnostic plots for a pipeline step."""
        plot_dir.mkdir(parents=True, exist_ok=True)
        cols = current_dict.columns

        if "x" in cols and "y" in cols:
            out = str(plot_dir / f"last_frame_{step_name}.jpg")
            if self.is_image_sequence:
                draw_results_on_last_frame(
                    current_dict, image_sequence=self.images, output_path=out
                )
            else:
                draw_results_on_last_frame(
                    current_dict, video_path=str(self.path), output_path=out
                )
            plot_targets_coordinates(
                current_dict,
                output_path=str(plot_dir / f"targets_coords_{step_name}.jpg"),
            )

        if all(c in cols for c in ["rvec_x", "rvec_y", "rvec_z", "tvec_x", "tvec_y", "tvec_z"]):
            plot_attitude(
                current_dict,
                rolling_window=4,
                output_path=str(plot_dir / f"attitude_{step_name}.jpg"),
            )

        if all(c in cols for c in ["drone_E", "drone_N", "drone_U", "tvec_E", "tvec_N", "tvec_U", "time"]):
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
        mode = "image_sequence" if self.is_image_sequence else "video"
        model = self.data[1] if self.data is not None else "not loaded"
        return f"Camera(path={self.path}, mode={mode}, model={model})"