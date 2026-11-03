"""
BOCK analysis: self-calibrating EKF for the camera attitude, intrinsics and the
IMU -> camera rotation.

Runs BOCK's error-state EKF (``ARS/BOCK/bock/ekf_filter.py``) on a pils ``Flight``,
with the camera preset (filter config, R_CB, priors) of ``BOCK/config/bockConfig.yaml``
for that camera (``sony`` / ``alvium``):

- prediction : IMX5 raw IMU (gyro), accelerometer update for gravity
- update     : the target PIXEL detections of a photogrammetry version (one row
               per target per frame, with the camera GNSS position), not the PnP
               quaternion -- so fx, fy, cx, cy, k1..k3, p1, p2 and R_CB are
               estimated as states, with covariance.

The Rust EKF (``EKFAnalysis``, PnP quaternion as update) is a separate analysis.
Both run on the same time base: the photogrammetry frame times are already on the
payload GPS clock (``PhotogrammetryAnalysis`` + ``flight.sync()``), which is the
clock of the IMX5 timestamps.

Results are stored per flight as versions, like the other analyses::

    flight_dir/proc/bock/
    ├── plots/
    ├── error_budget/rev_*/   (run_error_budget: MCMC-seeded EKF + camera systematic tables)
    └── bock_solution.h5      (one group per version)

This is the only pils module that imports BOCK (``ARS`` must be on ``sys.path``,
as for IPA_flight).
"""

import datetime
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
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

AXES = ["roll", "pitch", "yaw"]


@dataclass
class BOCKVersion:
    """BOCK analysis version container.

    Parameters
    ----------
    version_name : str
        Name identifier for this version (``rev_YYYYMMDD_HHMMSS``).
    states : pl.DataFrame
        Filter states over time (``kind`` = "photo" at each frame, "imu" every
        ``log_every_imu`` samples): quaternion, IMU and camera roll/pitch/yaw
        (FRD in NED, same as dictionary_p5) with 1-sigma, gyro bias, intrinsics.
    calibration : pl.DataFrame
        Final calibration: param, prior, estimate, sigma, shift_sigma.
    epochs : pl.DataFrame
        Photo update diagnostics per frame (targets used, NIS, pixel RMS).
    metadata : dict[str, Any]
        Inputs, filter config, final R_CB, gyro bias and calibration correlation.
    """

    version_name: str
    states: pl.DataFrame
    calibration: pl.DataFrame
    epochs: pl.DataFrame
    metadata: dict[str, Any]


class BOCKAnalysis:
    """Self-calibrating EKF (BOCK) for one flight.

    Examples
    --------
    >>> flight = Flight(flight_info)
    >>> flight.add_sensor_data("inclinometer")       # IMX5
    >>> bock = BOCKAnalysis(flight)
    >>> version = bock.run_analysis()                # latest photogrammetry version
    >>> version.calibration
    >>> stats, per_frame = bock.compare_with_photogrammetry(version)
    >>> # later, without re-running anything
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

    def _imu_df(self) -> pd.DataFrame:
        """IMX5 raw IMU as BOCK expects it: t [s], gx, gy, gz [rad/s], ax, ay, az [m/s^2]."""
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
    def _photo_df(photo_version: PhotogrammetryVersion) -> pd.DataFrame:
        """One row per detected target per frame: t, frame, target_id, x, y,
        target_E/N/U, cam_E/N/U (the drone GNSS position at the frame time)."""
        cols = ["frame", "target_id", "x", "y", "target_E", "target_N", "target_U",
                "time", "drone_E", "drone_N", "drone_U"]
        data = photo_version.photogrammetry_data
        missing = [c for c in cols if c not in data.columns]
        if missing:
            raise ValueError(
                f"Photogrammetry version '{photo_version.version_name}' has no {missing} "
                "(the pipeline must have reached p4)."
            )
        floats = [c for c in cols if c not in ("frame", "target_id")]
        df = (
            data.select(cols)
            .drop_nulls()
            .filter(pl.all_horizontal([pl.col(c).cast(pl.Float64).is_finite() for c in floats]))
            .unique(subset=["frame", "target_id"])      # rows repeat per telescope
            .sort(["time", "target_id"])
            .rename({"time": "t", "drone_E": "cam_E", "drone_N": "cam_N", "drone_U": "cam_U"})
            .to_pandas()
        )
        logger.info(f"Photo: {df.frame.nunique()} frames, {len(df)} target detections")
        return df

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
    # Error budget (BOCK/bock/error_budget.py)
    # ------------------------------------------------------------------

    def error_budget_inputs(
        self,
        photo_version: PhotogrammetryVersion | str | None = None,
        camera: str | None = None,
    ) -> SimpleNamespace:
        """Inputs of ``bock.error_budget.run`` from this flight.

        Returns a namespace with ``camera``, ``imu`` (IMX5 raw IMU), ``photo`` (one row
        per frame x target, canonical columns, rvec = GPS-corrected PnP), ``raw_pnp``
        (raw PnP pose, for the photo cleaning), ``intrinsics`` (camera block of the
        photogrammetry config = nominal camera) and ``photo_version``.
        """
        from BOCK.bock import error_budget as EB

        photo_version = self._resolve_photo_version(photo_version)
        data = photo_version.photogrammetry_data.to_pandas()
        return SimpleNamespace(
            camera=camera or self._camera_of(photo_version),
            imu=self._imu_df(),
            photo=EB.attitude_from_df(data),
            raw_pnp=data[EB.RAW_PNP_COLUMNS].dropna().drop_duplicates(["frame", "target_id"]),
            intrinsics={k: float(v) for k, v in self._intrinsics(photo_version).items()},
            photo_version=photo_version,
        )

    def run_error_budget(
        self,
        photo_version: PhotogrammetryVersion | str | None = None,
        camera: str | None = None,
        config_file: str | Path | None = None,
        run_systematic: bool = True,
        ekf_overrides: dict[str, Any] | None = None,
        max_seconds: float | None = None,
        save_data: bool = True,
        verbose: bool = True,
    ) -> SimpleNamespace:
        """MCMC-seeded EKF + camera systematic (``bock.error_budget.run``) on this flight.

        Settings: the ``error_budget`` block of the camera in the BOCK config. With
        ``save_data`` the tables go to ``proc/bock/error_budget/rev_YYYYMMDD_HHMMSS/``
        (``result.out_dir``).
        """
        from BOCK.bock import error_budget as EB

        inp = self.error_budget_inputs(photo_version, camera)
        res = EB.run(inp.camera, inp.imu, inp.photo, inp.intrinsics, raw_pnp=inp.raw_pnp,
                     config_file=config_file, run_systematic=run_systematic,
                     ekf_overrides=ekf_overrides, max_seconds=max_seconds, verbose=verbose)
        res.photo_version = inp.photo_version.version_name
        res.out_dir = None
        if save_data:
            res.out_dir = (self.bock_dir / "error_budget"
                           / datetime.datetime.now().strftime("rev_%Y%m%d_%H%M%S"))
            EB.save(res, res.out_dir)
        return res

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run_analysis(
        self,
        photo_version: PhotogrammetryVersion | str | None = None,
        camera: str | None = None,
        config_file: str | Path | None = None,
        frame_rotation_zyx_deg: list[float] | None = None,
        intrinsics: dict[str, float] | None = None,
        intrinsics_sigma: dict[str, float] | None = None,
        ekf_config: dict[str, Any] | None = None,
        bad_targets: list[int] | None = None,
        bad_target_frames: dict[int, tuple[int, int]] | None = None,
        max_seconds: float | None = None,
        save_data: bool = True,
        verbose: bool = True,
    ) -> BOCKVersion:
        """Run the self-calibrating EKF on this flight.

        Every prior defaults to the camera preset of the BOCK config
        (``BOCK/config/bockConfig.yaml``); the arguments override it.

        Parameters
        ----------
        photo_version : PhotogrammetryVersion | str | None
            Photogrammetry result to use (object or version name).
            None -> latest stored version.
        camera : str | None
            Preset ``"sony"`` or ``"alvium"``. None -> camera of the photogrammetry version.
        config_file : str | Path | None
            BOCK YAML config. None -> ``BOCK/config/bockConfig.yaml``.
        frame_rotation_zyx_deg : list[float] | None
            Prior IMU -> camera rotation R_CB, ZYX [z, y, x] deg.
        intrinsics : dict | None
            Prior fx, fy, cx, cy, k1, k2, p1, p2, k3.
            None -> camera block of the photogrammetry config.
        intrinsics_sigma : dict | None
            Prior 1-sigma, same keys (missing keys keep the preset value).
        ekf_config : dict | None
            Filter config fields on top of the preset
            (e.g. ``{"estimate_extrinsic": False, "camera_time_offset": 0.0}``).
        bad_targets : list[int] | None
            target_id dropped in every frame (mislabelled detections).
        bad_target_frames : dict[int, (int, int)] | None
            target_id dropped in a frame range (inclusive), e.g. ``{6: (406, 524)}``.
        max_seconds : float | None
            Process only the first N s of photos (quick tuning runs).
        save_data : bool
            Store the result as a new version in ``proc/bock/bock_solution.h5``.
        """
        from scipy.spatial.transform import Rotation

        from BOCK.bock import ekf_filter as E

        photo_version = self._resolve_photo_version(photo_version)
        config_file = Path(config_file or E.DEFAULT_CONFIG)
        presets = E.load_presets(config_file)
        camera = camera or self._camera_of(photo_version)
        if camera not in presets:
            raise ValueError(f"No BOCK preset for camera {camera!r} in {config_file} -- "
                             f"pass camera= one of {list(presets)}")
        preset = presets[camera]
        cfg = E.make_cfg(camera, config_file, **(ekf_config or {}))

        photo_df = self._photo_df(photo_version)
        bad = photo_df["target_id"].isin(bad_targets or [])
        for tid, (f0, f1) in (bad_target_frames or {}).items():
            bad |= (photo_df["target_id"] == tid) & photo_df["frame"].between(f0, f1)
        photo_df = photo_df[~bad].reset_index(drop=True)
        photo_df["t"] += cfg.camera_time_offset
        if max_seconds is not None:
            photo_df = photo_df[photo_df.t <= photo_df.t.min() + max_seconds]
        imu_df = self._imu_df()

        K = intrinsics or self._intrinsics(photo_version)
        S = {**preset["intrinsics_sigma"], **(intrinsics_sigma or {})}
        frame_rot = frame_rotation_zyx_deg or preset["frame_rotation_zyx_deg"]
        R_CB0 = E.R_CB_from_config(frame_rot)

        logger.info(f"BOCK ({camera}): photo version {photo_version.version_name}, R_CB0 (ZYX deg) {frame_rot}")
        states, calib, epochs, state = E.run_ekf(imu_df, photo_df, K, S, R_CB0, cfg=cfg, verbose=verbose)

        e = epochs.dropna(subset=["nis_per_dof"]) if "nis_per_dof" in epochs else epochs.iloc[:0]
        metadata = {
            "flight_name": self.flight_path.name,
            "camera": camera,
            "config_file": str(config_file),
            "photo_version": photo_version.version_name,
            "photo_config_file": photo_version.metadata.get("config_file"),
            "frame_rotation_zyx_deg_prior": list(frame_rot),
            "frame_rotation_zyx_deg_final": Rotation.from_matrix(state["R_CB"]).as_euler("ZYX", degrees=True).tolist(),
            "R_CB_final": state["R_CB"].tolist(),
            "sigma_dphi_deg": calib.attrs["sigma_dphi_deg"].tolist(),
            "gyro_bias_deg_s": np.degrees(state["bg"]).tolist(),
            "intrinsics_prior": {k: float(v) for k, v in K.items()},
            "intrinsics_sigma_prior": S,
            "ekf_config": vars(cfg),
            "bad_targets": bad_targets,
            "bad_target_frames": bad_target_frames,
            "calibration_corr": {
                "names": list(calib.attrs["corr"].columns),
                "matrix": calib.attrs["corr"].to_numpy().tolist(),
            },
            "max_seconds": max_seconds,
            "num_imu_samples": len(imu_df),
            "num_frames": int(photo_df.frame.nunique()),
            "num_photo_epochs_used": len(e),
            "nis_per_dof_median": float(e.nis_per_dof.median()) if len(e) else None,
            "rms_px_median": float(e.rms_px.median()) if len(e) else None,
        }

        version = BOCKVersion(
            version_name=datetime.datetime.now().strftime("rev_%Y%m%d_%H%M%S"),
            states=pl.from_pandas(states),
            calibration=pl.from_pandas(calib.reset_index()),
            epochs=pl.from_pandas(epochs),
            metadata=metadata,
        )
        if save_data:
            self._save_to_hdf5(version)
        return version

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def compare_with_photogrammetry(
        self,
        version: BOCKVersion,
        photo_version: PhotogrammetryVersion | None = None,
        skip_seconds: float = 10.0,
        plot: bool = True,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """EKF camera attitude vs the PnP attitude of the photogrammetry version
        (``yaw/pitch/roll[deg]`` of dictionary_p5, FRD in NED), per frame.

        Returns (stats per axis, per-frame table); ``skip_seconds`` drops the EKF
        start-up transient from the stats. The plot is saved to ``proc/bock/plots/``.
        """
        if photo_version is None:
            photo_version = PhotogrammetryAnalysis(self.flight).load_version(version.metadata["photo_version"])

        att_cols = [f"{a}[deg]" for a in AXES]
        p5 = (
            photo_version.photogrammetry_data
            .select(["frame", *att_cols, "projection_error"])
            .drop_nulls(subset=att_cols)
            .unique(subset=["frame"])
            .to_pandas()
        )
        # photo states <-> frame through the epochs (same t); t includes camera_time_offset
        st = version.states.filter(pl.col("kind") == "photo").to_pandas()
        st = st.merge(version.epochs.select(["t", "frame"]).to_pandas(), on="t")
        m = st.merge(p5, on="frame").sort_values("t").reset_index(drop=True)
        if m.empty:
            raise ValueError("No frame in common between the EKF states and the photogrammetry version.")
        m["trel"] = m.t - m.t.iloc[0]
        for a in AXES:
            m[f"d_{a}"] = _wrap_deg(m[f"{a}_cam_deg"] - m[f"{a}[deg]"])
        s = m[m.trel > skip_seconds]

        rows = []
        for a in AXES:
            d = s[f"d_{a}"]
            hf = d - d.rolling(31, center=True, min_periods=5).median()   # ~1 s high-pass
            rows.append({
                "axis": a,
                "mean_diff_deg": d.mean(), "median_diff_deg": d.median(), "sd_diff_deg": d.std(),
                "sd_diff_highpass_deg": hf.std(),
                "p99_abs_diff_deg": d.abs().quantile(0.99), "max_abs_diff_deg": d.abs().max(),
                "ekf_sigma_median_deg": s[f"sig_{a}_cam_deg"].median(),
            })
        stats = pd.DataFrame(rows)

        if plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(3, 2, figsize=(16, 11), sharex=True)
            for i, a in enumerate(AXES):
                ax[i, 0].plot(m.trel, np.degrees(np.unwrap(np.radians(m[f"{a}[deg]"]))), lw=0.6, label="PnP (p5)")
                ax[i, 0].plot(m.trel, np.degrees(np.unwrap(np.radians(m[f"{a}_cam_deg"]))), lw=0.6, label="EKF self-cal")
                ax[i, 0].set_ylabel(f"{a} [deg]")
                ax[i, 0].legend(loc="best")
                d, sg = m[f"d_{a}"], m[f"sig_{a}_cam_deg"]
                ax[i, 1].plot(m.trel, d, lw=0.4, color="k", label="EKF − PnP")
                ax[i, 1].fill_between(m.trel, d.median() - sg, d.median() + sg, alpha=0.3,
                                      label="± EKF 1σ (around median)")
                ax[i, 1].axhline(d.median(), color="r", ls="--", lw=0.8)
                if len(s):
                    lo, hi = s[f"d_{a}"].quantile([0.005, 0.995])
                    pad = 0.2 * (hi - lo)
                    ax[i, 1].set_ylim(lo - pad, hi + pad)
                r = stats.iloc[i]
                ax[i, 1].set_title(f"{a}: median {r.median_diff_deg:+.3f}°, sd {r.sd_diff_deg:.3f}°, "
                                   f"EKF σ {r.ekf_sigma_median_deg:.3f}°")
                ax[i, 1].legend(loc="best", fontsize=8)
            ax[-1, 0].set_xlabel("time from first frame [s]")
            ax[-1, 1].set_xlabel("time from first frame [s]")
            fig.tight_layout()
            self._save_fig(fig, f"{version.version_name}_compare_p5.png")

        return pl.from_pandas(stats), pl.from_pandas(m)

    def plot(self, version: BOCKVersion):
        """Camera attitude and its sigma, fx/cx/cy convergence, NIS, pixel RMS and gyro bias."""
        import matplotlib.pyplot as plt

        ph = version.states.filter(pl.col("kind") == "photo").to_pandas()
        calib = version.calibration.to_pandas().set_index("param")
        epochs = version.epochs.to_pandas()
        t = ph.t - ph.t.iloc[0]

        fig, ax = plt.subplots(4, 3, figsize=(16, 13))
        for c, a in enumerate(AXES):
            ax[0, c].plot(t, np.degrees(np.unwrap(np.radians(ph[f"{a}_cam_deg"]))), lw=0.6)
            ax[0, c].set_title(f"camera {a} [deg]")
            ax[1, c].semilogy(t, ph[f"sig_{a}_cam_deg"], lw=0.8)
            ax[1, c].set_title(f"sigma camera {a} [deg]")
        for c, k in enumerate(["fx", "cx", "cy"]):
            ax[2, c].plot(t, ph[k], lw=0.8)
            ax[2, c].fill_between(t, ph[k] - ph[f"sig_{k}"], ph[k] + ph[f"sig_{k}"], alpha=0.3)
            ax[2, c].set_title(f"{k} [px]  final {calib.loc[k, 'estimate']:.2f} ± {calib.loc[k, 'sigma']:.2f}")
        if "nis_per_dof" in epochs:
            e = epochs.dropna(subset=["nis_per_dof"])
            te = e.t - ph.t.iloc[0]
            ax[3, 0].plot(te, e.nis_per_dof, lw=0.4)
            ax[3, 1].plot(te, e.rms_px, lw=0.4)
        ax[3, 0].axhline(1, color="k", ls="--")
        ax[3, 0].set_title("photo NIS / dof (expect ~1)")
        ax[3, 1].set_title("pixel residual RMS [px]")
        for k in "xyz":
            ax[3, 2].plot(t, np.degrees(ph[f"bg_{k}"]), lw=0.8, label=k)
        ax[3, 2].set_title("gyro bias [deg/s]")
        ax[3, 2].legend()
        for a in ax[-1]:
            a.set_xlabel("time from first frame [s]")
        fig.tight_layout()
        self._save_fig(fig, f"{version.version_name}_bock.png")
        return fig

    def _save_fig(self, fig, name: str) -> None:
        plot_dir = self.bock_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_dir / name, dpi=110)
        logger.info(f"Plot: {plot_dir / name}")

    # ------------------------------------------------------------------
    # Versions (HDF5)
    # ------------------------------------------------------------------

    _TABLES = ("states", "calibration", "epochs")

    @property
    def _hdf5_path(self) -> Path:
        return self.bock_dir / "bock_solution.h5"

    def _save_to_hdf5(self, version: BOCKVersion) -> None:
        """Store a version: each table as parquet bytes (keeps every dtype) + JSON metadata."""
        self.bock_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(self._hdf5_path, "a") as f:
            if version.version_name in f:
                del f[version.version_name]
            group = f.create_group(version.version_name)
            for name in self._TABLES:
                buf = io.BytesIO()
                getattr(version, name).write_parquet(buf)
                group.create_dataset(name, data=np.frombuffer(buf.getvalue(), dtype=np.uint8))
            group.attrs["metadata"] = json.dumps(version.metadata, default=str)
        logger.info(f"Saved BOCK version '{version.version_name}' to {self._hdf5_path}")

    def load_version(self, version_name: str) -> BOCKVersion:
        if not self._hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self._hdf5_path}")
        with h5py.File(self._hdf5_path, "r") as f:
            if version_name not in f:
                raise KeyError(f"Version '{version_name}' not found in {self._hdf5_path}")
            group = f[version_name]
            tables = {name: pl.read_parquet(io.BytesIO(group[name][()].tobytes())) for name in self._TABLES}
            metadata = json.loads(group.attrs["metadata"])
        return BOCKVersion(version_name, metadata=metadata, **tables)

    def list_versions(self) -> list[str]:
        if not self._hdf5_path.exists():
            return []
        with h5py.File(self._hdf5_path, "r") as f:
            return sorted(f.keys())

    def get_latest_version(self) -> BOCKVersion | None:
        versions = self.list_versions()
        return self.load_version(versions[-1]) if versions else None


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0
