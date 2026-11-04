"""
BOCK analysis: inputs and outputs of the BOCK attitude error budget (``ARS/BOCK``).

BOCK (``BOCK/bock/mcmc.py``, ``BOCK/bock/ekf.py``, settings ``BOCK/config/bockConfig.yaml``,
notebook ``BOCK/examples/bock.ipynb``) does the computation:

1. MCMC: joint multi-frame fit of the camera calibration + camera systematic (Option A)
2. self-calibrating EKF of the attitude seeded with the MCMC posterior

This module only handles the I/O on a pils ``Flight``:

- ``load_inputs``: IMX5 raw IMU, target detections + GNSS camera position + PnP poses of a
  photogrammetry version (one row per frame x target, canonical columns), raw PnP poses,
  nominal camera (camera block of the photogrammetry config) and the camera preset name;
- ``save_version`` / ``load_version`` / ``list_versions``: result tables, metadata and
  figures of a run, one directory per version::

    flight_dir/proc/bock/
    └── rev_YYYYMMDD_HHMMSS/
        ├── metadata.json
        ├── <table>.csv | <table>.parquet
        └── figures/<name>.png

The photogrammetry frame times are on the payload GPS clock (``PhotogrammetryAnalysis`` +
``flight.sync()``), the clock of the IMX5 timestamps; the residual IMU <-> camera offset is
estimated by BOCK.
"""

import datetime
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from pils.analyze.photogrammetry import (
    PhotogrammetryAnalysis,
    PhotogrammetryConfig,
    PhotogrammetryVersion,
)
from pils.flight import Flight

logger = logging.getLogger(__name__)

# canonical name -> accepted photogrammetry column names (first present wins); the IPA_flight output and
# the PHOTOGRAMMETRY_RESULTS output name a few columns differently (time vs time_x, drone_E vs drone_E_x)
PHOTO_COLUMNS = {
    "frame": ["frame"], "target_id": ["target_id"], "x": ["x"], "y": ["y"],
    "target_E": ["target_E"], "target_N": ["target_N"], "target_U": ["target_U"],
    "t":     ["time", "time_x", "time_y"],
    "cam_E": ["drone_E", "drone_E_x", "drone_E_y"],
    "cam_N": ["drone_N", "drone_N_x", "drone_N_y"],
    "cam_U": ["drone_U", "drone_U_x", "drone_U_y"],
    "rvec_x": ["rvec_x_corr"], "rvec_y": ["rvec_y_corr"], "rvec_z": ["rvec_z_corr"],
    "pnp_E": ["tvec_E"], "pnp_N": ["tvec_N"], "pnp_U": ["tvec_U"],
}
RAW_PNP_COLUMNS = ["frame", "target_id", "rvec_x", "rvec_y", "rvec_z", "tvec_x", "tvec_y", "tvec_z"]


@dataclass
class BOCKVersion:
    """A stored BOCK run.

    Parameters
    ----------
    version_name : str
        ``rev_YYYYMMDD_HHMMSS``.
    path : Path
        Version directory.
    tables : dict[str, pd.DataFrame]
        Result tables, by name (file stem).
    metadata : dict[str, Any]
        Inputs, settings and scalar results of the run.
    """

    version_name: str
    path: Path
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BOCKAnalysis:
    """I/O of the BOCK attitude error budget for one flight.

    Examples
    --------
    >>> flight = Flight(flight_info)
    >>> flight.add_sensor_data("inclinometer")       # IMX5
    >>> bock = BOCKAnalysis(flight)
    >>> inp = bock.load_inputs()                     # latest photogrammetry version
    >>> ...                                          # BOCK.bock.mcmc / BOCK.bock.ekf
    >>> bock.save_version(tables, metadata, figures)
    >>> BOCKAnalysis(flight).get_latest_version()
    """

    def __init__(self, flight: Flight) -> None:
        if not isinstance(flight, Flight):
            raise TypeError(
                f"Expected Flight object, got {type(flight).__name__}. "
                "BOCKAnalysis requires a Flight object."
            )
        if flight.flight_path is None:
            raise ValueError("Flight object must have a valid flight_path attribute.")

        flight_path = Path(flight.flight_path)
        if not flight_path.exists() or not flight_path.is_dir():
            raise ValueError(f"flight_path must be an existing directory. Got: {flight_path}")

        self.flight = flight
        self.flight_path = flight_path
        self.bock_dir = self.flight_path / "proc" / "bock"

        logger.info(f"Initialized BOCK analysis for flight: {self.flight_path}")

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def load_inputs(
        self,
        photo_version: PhotogrammetryVersion | str | None = None,
        camera: str | None = None,
        bad_targets: list[int] | None = None,
        bad_target_frames: dict[int, tuple[int, int]] | None = None,
        max_seconds: float | None = None,
    ) -> SimpleNamespace:
        """Inputs of BOCK from this flight.

        Parameters
        ----------
        photo_version : PhotogrammetryVersion | str | None
            Photogrammetry result (object or version name). None -> latest stored version.
        camera : str | None
            Camera preset of the BOCK config ("sony" / "alvium"). None -> camera of the
            photogrammetry version.
        bad_targets : list[int] | None
            target_id dropped in every frame (mislabelled detections).
        bad_target_frames : dict[int, (int, int)] | None
            target_id dropped in a frame range (inclusive), e.g. ``{6: (406, 524)}``.
        max_seconds : float | None
            Keep only the first N s of photo frames.

        Returns
        -------
        SimpleNamespace
            ``camera``, ``imu`` (t, gx..gz [rad/s], ax..az [m/s^2]), ``photo`` (one row per
            frame x target: frame, target_id, x, y, target_E/N/U, t, cam_E/N/U, rvec_x/y/z =
            GPS-corrected PnP, pnp_E/N/U), ``raw_pnp`` (raw PnP pose per frame x target),
            ``intrinsics`` (nominal camera fx..k3), ``photo_version``.
        """
        photo_version = self._resolve_photo_version(photo_version)
        data = photo_version.photogrammetry_data.to_pandas()

        photo = self._photo_df(data)
        bad = photo["target_id"].isin(bad_targets or [])
        for tid, (f0, f1) in (bad_target_frames or {}).items():
            bad |= (photo["target_id"] == tid) & photo["frame"].between(f0, f1)
        photo = photo[~bad].reset_index(drop=True)
        if max_seconds is not None:
            photo = photo[photo["t"] <= photo["t"].min() + max_seconds].reset_index(drop=True)
        logger.info(f"Photo: {photo.frame.nunique()} frames, {len(photo)} target detections")

        missing = [c for c in RAW_PNP_COLUMNS if c not in data.columns]
        raw_pnp = (None if missing else
                   data[RAW_PNP_COLUMNS].dropna().drop_duplicates(["frame", "target_id"]).reset_index(drop=True))

        camera = camera or self._camera_of(photo_version)
        if camera is None:
            raise ValueError("Camera of the photogrammetry version unknown: pass camera='sony' / 'alvium'.")
        return SimpleNamespace(
            camera=camera,
            imu=self._imu_df(),
            photo=photo,
            raw_pnp=raw_pnp,
            intrinsics={k: float(v) for k, v in self._intrinsics(photo_version).items()},
            photo_version=photo_version,
        )

    def _imu_df(self) -> pd.DataFrame:
        """IMX5 raw IMU: t [s], gx, gy, gz [rad/s], ax, ay, az [m/s^2]."""
        payload = self.flight.raw_data.payload_data
        incl = getattr(payload, "inclinometer", None)
        if not isinstance(incl, dict) or "IMU" not in incl:
            self.flight.add_sensor_data("inclinometer")
            incl = getattr(payload, "inclinometer", None)
        if not isinstance(incl, dict) or "IMU" not in incl:
            raise ValueError("No IMX5 raw IMU (aux/sensors/*_imu.csv) for this flight.")

        cols = ["timestamp_ns", "pqr_P_rad_s", "pqr_Q_rad_s", "pqr_R_rad_s",
                "acc_X_m_s2", "acc_Y_m_s2", "acc_Z_m_s2"]
        # a truncated last line can leave string columns -> cast non-strict, drop non-finite
        raw = incl["IMU"].select(
            [pl.col(c).cast(pl.Float64, strict=False) for c in cols]
        ).drop_nulls()
        raw = raw.filter(
            pl.all_horizontal([pl.col(c).is_finite() for c in cols])
            & (pl.col("timestamp_ns") > 1.5e18)            # corrupted timestamps (before 2017)
        )
        imu = pd.DataFrame({
            "t": raw["timestamp_ns"].to_numpy() / 1e9,
            "gx": raw["pqr_P_rad_s"].to_numpy(), "gy": raw["pqr_Q_rad_s"].to_numpy(),
            "gz": raw["pqr_R_rad_s"].to_numpy(),
            "ax": raw["acc_X_m_s2"].to_numpy(), "ay": raw["acc_Y_m_s2"].to_numpy(),
            "az": raw["acc_Z_m_s2"].to_numpy(),
        }).sort_values("t").reset_index(drop=True)
        logger.info(f"IMU: {len(imu)} samples, t = {imu.t.min():.3f} - {imu.t.max():.3f}")
        return imu

    @staticmethod
    def _photo_df(data: pd.DataFrame) -> pd.DataFrame:
        """Photogrammetry result -> one row per (frame, target), canonical column names."""
        pick = {}
        for k, cands in PHOTO_COLUMNS.items():
            src = next((c for c in cands if c in data.columns), None)
            if src is None:
                raise ValueError(f"Photogrammetry result has none of {cands} (the pipeline must have reached p4).")
            pick[k] = src
        return (data[list(pick.values())]
                .rename(columns={v: k for k, v in pick.items()})
                .drop_duplicates(["frame", "target_id"])          # rows repeat per telescope
                .dropna()
                .sort_values(["t", "target_id"])
                .reset_index(drop=True))

    @staticmethod
    def _intrinsics(photo_version: PhotogrammetryVersion) -> dict[str, float]:
        """fx..k3 from the camera block of the config the photogrammetry version ran with."""
        meta = photo_version.metadata
        cfg = PhotogrammetryConfig(meta["config_file"], camera_model=meta.get("camera_model"))
        K = cfg.camera_matrix
        D = np.zeros(5)
        d = np.asarray(cfg.distortion_coeffs, dtype=float).ravel()[:5]
        D[: len(d)] = d
        return dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                    k1=D[0], k2=D[1], p1=D[2], p2=D[3], k3=D[4])

    @staticmethod
    def _camera_of(photo_version: PhotogrammetryVersion) -> str | None:
        """'sony' / 'alvium' from the camera model the photogrammetry version ran with."""
        model = str(photo_version.metadata.get("camera_model") or "").lower()
        return next((c for c in ("sony", "alvium") if c in model), None)

    def _resolve_photo_version(
        self, photo_version: PhotogrammetryVersion | str | None
    ) -> PhotogrammetryVersion:
        """Object, version name, or None (latest stored version)."""
        if photo_version is None or isinstance(photo_version, str):
            photo = PhotogrammetryAnalysis(self.flight)
            photo_version = (photo.get_latest_version() if photo_version is None
                             else photo.load_version(photo_version))
            if photo_version is None:
                raise FileNotFoundError(
                    f"No photogrammetry version in {photo.photogrammetry_dir} -- run "
                    "PhotogrammetryAnalysis.run_analysis first."
                )
        return photo_version

    # ------------------------------------------------------------------
    # Outputs (versions)
    # ------------------------------------------------------------------

    def save_version(
        self,
        tables: dict[str, pd.DataFrame],
        metadata: dict[str, Any] | None = None,
        figures: dict[str, Any] | None = None,
        version_name: str | None = None,
    ) -> BOCKVersion:
        """Store a run as a new version in ``proc/bock/rev_YYYYMMDD_HHMMSS/``.

        Parameters
        ----------
        tables : dict[str, pd.DataFrame]
            Name -> table. A name ending in ``.parquet`` is written as parquet, anything else as
            CSV (``.csv`` added). Named indexes are kept.
        metadata : dict | None
            JSON-serialisable run information (numpy values are converted).
        figures : dict[str, matplotlib.figure.Figure] | None
            Name -> figure, saved as ``figures/<name>.png``.
        """
        version_name = version_name or datetime.datetime.now().strftime("rev_%Y%m%d_%H%M%S")
        path = self.bock_dir / version_name
        path.mkdir(parents=True, exist_ok=True)
        stored = {}
        for name, df in tables.items():
            stem = name.removesuffix(".parquet").removesuffix(".csv")
            keep_index = any(n is not None for n in df.index.names)
            if name.endswith(".parquet"):
                df.to_parquet(path / f"{stem}.parquet", index=keep_index)
            else:
                df.to_csv(path / f"{stem}.csv", index=keep_index)
            stored[stem] = df
        meta = {"version_name": version_name, "flight_name": self.flight_path.name, **(metadata or {})}
        (path / "metadata.json").write_text(json.dumps(meta, indent=1, default=_to_json))
        if figures:
            (path / "figures").mkdir(exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(path / "figures" / f"{name}.png", dpi=150, bbox_inches="tight")
        logger.info(f"Saved BOCK version '{version_name}' to {path}")
        return BOCKVersion(version_name, path, stored, meta)

    def list_versions(self) -> list[str]:
        if not self.bock_dir.exists():
            return []
        return sorted(p.name for p in self.bock_dir.glob("rev_*") if (p / "metadata.json").exists())

    def load_version(self, version_name: str) -> BOCKVersion:
        path = self.bock_dir / version_name
        if not (path / "metadata.json").exists():
            raise KeyError(f"Version '{version_name}' not found in {self.bock_dir}")
        tables = {}
        for f in sorted(path.iterdir()):
            if f.suffix == ".parquet":
                tables[f.stem] = pd.read_parquet(f)
            elif f.suffix == ".csv":
                tables[f.stem] = pd.read_csv(f)
        return BOCKVersion(version_name, path, tables, json.loads((path / "metadata.json").read_text()))

    def get_latest_version(self) -> BOCKVersion | None:
        versions = self.list_versions()
        return self.load_version(versions[-1]) if versions else None


def _to_json(v):
    """json default: numpy scalars / arrays, paths, anything else as str."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)
