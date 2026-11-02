"""
Photogrammetry analysis: drone attitude from the payload camera images.

Runs the IPA_flight pipeline (target tracking, PnP, drone GPS correlation,
polarization angles) on a pils ``Flight``, with every sensor on one time base:
before IPA_flight sees them, the drone and camera timestamps are shifted by
the offsets ``flight.sync()`` finds against the payload GPS.

Results are stored per flight as versions, like the other analyses::

    flight_dir/proc/photogrammetry/
    ├── dictionary_p2.ecsv ... dictionary_p6.ecsv   (pipeline checkpoints)
    ├── attitude_reconstruction.parquet
    ├── plots/
    └── photogrammetry_solution.h5                  (one group per version)

This is the only pils module that imports IPA_flight.
"""

import copy
import datetime
import io
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pandas as pd
import polars as pl

from pils.flight import Flight

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

        raw_lever_arm = pipeline.get("lever_arm_enu")
        self.lever_arm_enu = np.array(raw_lever_arm, dtype="double") if raw_lever_arm is not None else None

        self.finder = pipeline.get("finder", {})
        self.target_filter = pipeline.get("target_filter", {})
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


@dataclass
class PhotogrammetryVersion:
    """Photogrammetry analysis version container.

    Parameters
    ----------
    version_name : str
        Name identifier for this version (``rev_YYYYMMDD_HHMMSS``).
    photogrammetry_data : pl.DataFrame
        Merged pipeline result (steps p2-p6), one row per frame x telescope.
    metadata : dict[str, Any]
        Config, targets CSV, sync offsets and run options.
    """

    version_name: str
    photogrammetry_data: pl.DataFrame
    metadata: dict[str, Any]


class PhotogrammetryAnalysis:
    """Photogrammetry attitude reconstruction for one flight (IPA_flight pipeline).

    Examples
    --------
    >>> flight = Flight(flight_info)
    >>> flight.add_drone_data()
    >>> flight.add_camera_data(use_photogrammetry=False)
    >>> photo = PhotogrammetryAnalysis(flight)
    >>> version = photo.run_analysis(csv_file=targets_csv, config=PhotogrammetryConfig(cfg_yaml))
    >>> version.photogrammetry_data.head()
    >>> # later, without re-running anything
    >>> PhotogrammetryAnalysis(flight).get_latest_version()
    """

    def __init__(self, flight: Flight) -> None:
        if not isinstance(flight, Flight):
            raise TypeError(
                f"Expected Flight object, got {type(flight).__name__}. "
                "PhotogrammetryAnalysis requires a Flight object."
            )
        if flight.flight_path is None:
            raise ValueError("Flight object must have a valid flight_path attribute.")

        flight_path = Path(flight.flight_path)
        if not flight_path.exists() or not flight_path.is_dir():
            raise ValueError(f"flight_path must be an existing directory. Got: {flight_path}")

        self.flight = flight
        self.flight_path = flight_path
        self.photogrammetry_dir = self.flight_path / "proc" / "photogrammetry"

        logger.info(f"Initialized photogrammetry analysis for flight: {self.flight_path}")

    # ------------------------------------------------------------------
    # Time base (pils synchronizer)
    # ------------------------------------------------------------------

    def _synced_inputs(self, use_sync: bool = True):
        """
        Flight and camera with the drone, Litchi and camera timestamps shifted onto
        the payload GPS clock, using the offsets of ``flight.sync()``.

        The original data are not resampled (IPA_flight needs every drone fix and
        each camera frame's own timestamp) and the Flight passed in is not modified:
        shifted copies are returned.

        Returns
        -------
        (Flight, Camera, dict)
            Aligned flight, its aligned camera, and the offsets applied (s).
        """
        flight = self.flight
        camera = flight.raw_data.payload_data.camera_obj
        if not use_sync:
            return flight, camera, {}

        if flight.sync_offsets is None:
            payload = flight.raw_data.payload_data
            for sensor in ("gps", "inclinometer"):
                if getattr(payload, sensor, None) is None:
                    try:
                        flight.add_sensor_data(sensor)
                    except Exception as e:  # inclinometer is optional for the sync
                        logger.warning(f"Could not load payload {sensor} for the sync: {e}")
            flight.sync()

        offsets = flight.sync_offsets or {}
        if "drone_gps" not in offsets:
            raise RuntimeError(
                "flight.sync() found no drone GPS offset against the payload GPS "
                "(the drone and payload GPS tracks don't overlap in time -- e.g. a DJI "
                ".DAT whose clock was never set). Fix the drone clock first, or pass "
                "use_sync=False to keep the timestamps as they are."
            )
        drone_dt = float(offsets["drone_gps"]["time_offset"])
        litchi_dt = float(offsets.get("litchi_gps", {}).get("time_offset", 0.0))
        # same total as Synchronizer.synchronize: camera->Litchi + Litchi->GPS
        camera_dt = float(offsets.get("camera", {}).get("time_offset", 0.0)) + litchi_dt
        applied = {"drone": drone_dt, "litchi": litchi_dt, "camera": camera_dt}
        logger.info(f"Sync offsets applied (s): {applied}")

        dd = flight.raw_data.drone_data
        drone = dd.drone.with_columns(pl.col("timestamp") + drone_dt).with_columns(
            pl.from_epoch((pl.col("timestamp") * 1e6).round().cast(pl.Int64), time_unit="us").alias("datetime")
        )
        litchi = dd.litchi
        if litchi is not None and litchi.height > 0 and "datetime" in litchi.columns:
            litchi = litchi.with_columns(
                pl.col("datetime") + pl.duration(microseconds=round(litchi_dt * 1e6))
            )

        camera = copy.copy(camera)
        table, model = camera.data
        if table is not None and table.height > 0 and "timestamp" in table.columns:
            table = table.with_columns(pl.col("timestamp") + camera_dt)
        camera.data = (table, model)
        if camera.tstart is not None:
            camera.tstart = camera.tstart + datetime.timedelta(seconds=camera_dt)

        aligned = copy.copy(flight)
        aligned.raw_data = copy.copy(flight.raw_data)
        aligned.raw_data.drone_data = copy.copy(dd)
        aligned.raw_data.drone_data.drone = drone
        aligned.raw_data.drone_data.litchi = litchi
        aligned.raw_data.payload_data = copy.copy(flight.raw_data.payload_data)
        aligned.raw_data.payload_data.camera_obj = camera
        return aligned, camera, applied

    # ------------------------------------------------------------------
    # Internal: DataHandler wrapper builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_dh_wrapper(flight, camera_obj) -> SimpleNamespace:
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
    def run_multi_flights(
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

        Steps p3–p6 run independently per flight so outputs land in separate
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
            wrappers.append(PhotogrammetryAnalysis._build_dh_wrapper(flight, camera_obj))
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
            print(f"\n{'='*60}\n  Pipeline steps p3–p6: flight {orig_idx + 1}\n{'='*60}")

            # Slice this flight's tracking rows out of the combined DataFrame
            if all_targets_df is not None:
                flight_targets = all_targets_df[
                    all_targets_df["video_idx"] == list_idx
                ].drop(columns=["video_idx"])
            else:
                flight_targets = None  # run_photogrammetry will use start_from_dict

            try:
                result = PhotogrammetryAnalysis(flight).run_analysis(
                    csv_file=csv_file,
                    config=config,
                    output_dir=output_dir,
                    check_results=None,
                    start_from_dict=start_from_dict,
                    precomputed_targets=flight_targets,
                    mcmc_solution=mcmc_solution,
                    target_indices=target_indices,
                )
                result = result.photogrammetry_data.with_columns([pl.lit(orig_idx).alias("flight_idx")])
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

    def run_analysis(
        self,
        csv_file: str | Path,
        config: "PhotogrammetryConfig",
        output_dir: str | Path | None = None,
        gps_offset: list[float] | None = None,
        check_results: str | Path | None = None,
        start_from_dict: str | Path | None = None,
        precomputed_targets=None,
        target_indices: list[int] | None = None,
        mcmc_solution: bool = False,
        remove_by_eye: bool = True,
        use_sync: bool = True,
        save_data: bool = True,
    ) -> PhotogrammetryVersion:
        """Run the full IPA_flight photogrammetry pipeline for a single flight.

        Parameters
        ----------
        csv_file : str | Path
            Path to CSV with geodetic targets and telescope positions.
        config : PhotogrammetryConfig
            Loaded config carrying camera intrinsics and pipeline parameters.
        output_dir : str | Path | None
            None (default): ``<flight>/proc/photogrammetry/``. Otherwise a root
            folder, in which a per-flight subfolder is created (the layout of
            earlier runs, e.g. ``<day>/proc/<flight_name>/``).
        gps_offset : list[float] | None
            Optional RTK base correction ``[dlat, dlon, dalt]`` in geodetic
            degrees/metres (surveyed base minus DJI-logged base), added to
            the drone's raw lat/lon/alt before ENU conversion. Normally left
            as ``None`` -- it is computed from the CSV's logged vs. surveyed
            base rows and passed in automatically via ``params["gps_offset"]``
            (see ``GenParamFile.GeoPlot``); it is NOT an ENU-metres offset.
        check_results : str | Path | None
            If set, diagnostic plots are saved here.
            Defaults to ``<output_dir>/<flight_name>/plots/``.
        start_from_dict : str | Path | None
            Path to an intermediate ``.ecsv`` to resume from mid-pipeline.
        precomputed_targets : pd.DataFrame | pl.DataFrame | None
            When supplied by ``run_photogrammetry_multi_flights``, step p2
            (TargetFinder) is skipped and this DataFrame is used directly.
        remove_by_eye : bool
            When resuming from ``dictionary_p2.ecsv`` (``start_from_dict``
            pointing at it), pop up an interactive scatter of the last
            tracked frame after automated filtering runs, so any target
            none of ``filter_mixed_targets`` / ``filter_erratic_targets`` /
            ``filter_bad_detections`` caught (e.g. a track that smoothly,
            consistently latched onto the wrong-but-nearby blob the whole
            video) can be dropped by eye -- see
            ``TargetFinder.remove_targets_by_eye``. Set ``False`` for
            unattended/batch resumes. No effect on a fresh (non-resumed) run.
        use_sync : bool
            Shift the drone and camera timestamps onto the payload GPS clock
            with ``flight.sync()`` (loads the payload gps/inclinometer if needed)
            before running. False keeps each sensor's own timestamps.
        save_data : bool
            Store the result as a new version in ``photogrammetry_solution.h5``.

        Returns
        -------
        PhotogrammetryVersion
            Final merged photogrammetry result (``.photogrammetry_data``) + metadata.
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
        # from IPA_flight.IPA_flight.telescope import ConcerningTelescope
        from IPA_flight.IPA_flight.genParamFile import GenParamFile

        # ── Time base: drone + camera shifted onto the payload GPS clock ───
        flight, camera, sync_offsets = self._synced_inputs(use_sync)

        # ── Per-flight output subfolder ────────────────────────────────────
        flight_name = (
            flight.metadata.get("flight_name")
            or Path(flight.flight_info["drone_data_folder_path"]).parent.name
        )
        if output_dir is None:
            output_dir = self.photogrammetry_dir
        else:
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
        dh_wrapper = self._build_dh_wrapper(flight, camera)

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

        tel_names = config.telescope.get(
            "tel_names", ["SATp1", "SATp2", "SATp3", "CLASS1", "CLASS2"]
        )

        # telescope = ConcerningTelescope(
        #     params["geodetic_tel_positions"],
        #     params["reference_point"],
        #     tel_names=tel_names,
        # )

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
                prev, gps_offset=gps_offset,
                lever_arm_enu=getattr(config, "lever_arm_enu", None),
                **config.drone_correlation
            ),
            "dictionary_p5.ecsv": lambda prev: attitude_reconstruction._correct_tvec_with_gps(
                dataDict=prev,
                targetDict=load_dictionary(output_dir / "dictionary_p2.ecsv"),
            ),
            # "dictionary_p6.ecsv": lambda prev: telescope.drone_in_telescope_frame(prev),
            # "dictionary_p7.ecsv": lambda prev: polarization_angle.drone_attitude_in_LOS_frame(prev),
            "dictionary_p6.ecsv": lambda prev: polarization_angle.gen_yaw_pitch_roll_all_telescopes(
                prev,
                params["geodetic_tel_positions"],
                tel_names,
                **config.polarization,
            ),
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

            if start_file == "dictionary_p2.ecsv":
                raw_path = output_dir / "dictionary_p2_raw.ecsv"
                if not raw_path.exists():
                    # First time we've seen this flight's raw detections --
                    # the loaded dictionary_p2.ecsv is our only copy, keep it.
                    save_intermediate_results(
                        current_dict,
                        str(raw_path),
                        params["video_code"],
                        params["date"],
                        params["video_name"],
                    )
                else:
                    # dictionary_p2.ecsv on disk is already the *filtered*
                    # output of a prior run (the pipeline always persists it
                    # post-filter) -- never overwrite the true raw file with it.
                    logger.info(
                        f"{raw_path.name} already exists — leaving it untouched "
                        "(loaded dictionary_p2.ecsv is presumed already-filtered)."
                    )
                current_dict = self._ensure_polars(
                    target_finder.filter_targets(
                        current_dict.to_pandas(), **config.target_filter
                    )
                )
                if remove_by_eye:
                    print(
                        "\n👁  Manual review — RIGHT-click any target still wrong "
                        "after automated filtering, ENTER to confirm, close the "
                        "window to skip …"
                    )
                    current_dict = self._ensure_polars(
                        target_finder.remove_targets_by_eye(current_dict.to_pandas())
                    )
                # dictionary_p5.ecsv re-reads dictionary_p2.ecsv straight from
                # disk (see the step lambda above) rather than using
                # current_dict, so the on-disk file must stay in sync here too.
                save_intermediate_results(
                    current_dict,
                    str(output_dir / "dictionary_p2.ecsv"),
                    params["video_code"],
                    params["date"],
                    params["video_name"],
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

            if step_name == "dictionary_p2.ecsv":
                save_intermediate_results(
                    current_dict,
                    str(output_dir / "dictionary_p2_raw.ecsv"),
                    params["video_code"],
                    params["date"],
                    params["video_name"],
                )
                current_dict = self._ensure_polars(
                    target_finder.filter_targets(
                        current_dict.to_pandas(), **config.target_filter
                    )
                )

            save_intermediate_results(
                current_dict,
                str(output_dir / step_name),
                params["video_code"],
                params["date"],
                params["video_name"],
            )

            if check_results:
                self._run_check_results(
                    camera,
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
        # Several steps re-emit columns unchanged from the step before them
        # (e.g. p4 carries p3's rvec_* through untouched, p5 carries p4's
        # GPS fix through untouched). Joining on "frame" alone would otherwise
        # duplicate each of those as "<col>_right" (polars' default join
        # suffix) -- drop the redundant copy instead so each quantity appears
        # exactly once, under its one name. `coalesce=True` does the same for
        # the "frame" key itself.
        # p6 is now gen_yaw_pitch_roll_all_telescopes (one row per
        # frame x tel_name, replacing the old p6+p7 pair below) -- joining it
        # on "frame" alone broadcasts each frame's p2-p5 columns across all
        # of that frame's telescopes, same as the old p6 join did.
        p2, p3, p4, p5, p6 = [
            self._ensure_polars(
                load_dictionary(str(output_dir / f"dictionary_p{i}.ecsv"))
            )
            for i in range(2, 7)
        ]

        df_merged = (
            p2
            .join(p3, on="frame", how="full", coalesce=True)
            .join(
                p4.drop(["rvec_x", "rvec_y", "rvec_z"]),
                on="frame", how="full", coalesce=True,
            )
            .join(
                p5.drop(["time", "drone_E", "drone_N", "drone_U"]),
                on="frame", how="full", coalesce=True,
            )
            .join(p6.drop("time"), on="frame", how="full", coalesce=True)
        )

        out_parquet = output_dir / "attitude_reconstruction.parquet"
        df_merged.write_parquet(str(out_parquet))
        logger.info(f"Photogrammetry result saved: {out_parquet}")

        version = PhotogrammetryVersion(
            version_name=datetime.datetime.now().strftime("rev_%Y%m%d_%H%M%S"),
            photogrammetry_data=df_merged,
            metadata={
                "flight_name": flight_name,
                "config_file": str(config.config_path),
                "camera_model": camera.data[1] if camera.data else None,
                "csv_file": str(csv_file),
                "output_dir": str(output_dir),
                "sync_offsets_s": sync_offsets,
                "start_from_dict": str(start_from_dict) if start_from_dict else None,
                "target_indices": target_indices,
                "mcmc_solution": mcmc_solution,
                "num_rows": df_merged.height,
            },
        )
        if save_data:
            self._save_to_hdf5(version)
        return version

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_polars(d) -> pl.DataFrame:
        if isinstance(d, pd.DataFrame):
            return pl.from_pandas(d)
        return d

    @staticmethod
    def _run_check_results(
        camera,
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
            if camera.is_image_sequence:
                draw_results_on_last_frame(plot_df, image_sequence=camera.images, output_path=out)
            else:
                draw_results_on_last_frame(plot_df, video_path=str(camera.path), output_path=out)
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

    # ------------------------------------------------------------------
    # Versions (HDF5)
    # ------------------------------------------------------------------

    @property
    def _hdf5_path(self) -> Path:
        return self.photogrammetry_dir / "photogrammetry_solution.h5"

    def _save_to_hdf5(self, version: PhotogrammetryVersion) -> None:
        """Store a version: the result as parquet bytes (keeps every dtype) + JSON metadata."""
        self.photogrammetry_dir.mkdir(parents=True, exist_ok=True)
        buf = io.BytesIO()
        version.photogrammetry_data.write_parquet(buf)
        with h5py.File(self._hdf5_path, "a") as f:
            if version.version_name in f:
                del f[version.version_name]
            group = f.create_group(version.version_name)
            group.create_dataset("photogrammetry_data", data=np.frombuffer(buf.getvalue(), dtype=np.uint8))
            group.attrs["metadata"] = json.dumps(version.metadata, default=str)
        logger.info(f"Saved photogrammetry version '{version.version_name}' to {self._hdf5_path}")

    def load_version(self, version_name: str) -> PhotogrammetryVersion:
        if not self._hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self._hdf5_path}")
        with h5py.File(self._hdf5_path, "r") as f:
            if version_name not in f:
                raise KeyError(f"Version '{version_name}' not found in {self._hdf5_path}")
            group = f[version_name]
            data = pl.read_parquet(io.BytesIO(group["photogrammetry_data"][()].tobytes()))
            metadata = json.loads(group.attrs["metadata"])
        return PhotogrammetryVersion(version_name, data, metadata)

    def list_versions(self) -> list[str]:
        if not self._hdf5_path.exists():
            return []
        with h5py.File(self._hdf5_path, "r") as f:
            return sorted(f.keys())

    def get_latest_version(self) -> PhotogrammetryVersion | None:
        versions = self.list_versions()
        return self.load_version(versions[-1]) if versions else None
