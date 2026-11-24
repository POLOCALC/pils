"""
BOCK EKF attitude as input of the photogrammetry steps p6/p7, and the beam plots of the photogrammetry run.

- ``bock_ekf_to_p5``: BOCK ``ekf_states`` (IMU rate) -> a table in the format of IPA_flight ``dictionary_p5.ecsv``,
  so that p6 (``PolarizationAngle.gen_yaw_pitch_roll_all_telescopes``) and p7 (``GaussianBeam.project_p6``) run on
  the EKF attitude, i.e. an upsampled beam projection.
- ``plot_bock_ekf``: EKF camera attitude (±1σ) vs the p5 photogrammetry attitude.
- ``beam_image_map`` / ``plot_beam_image_map``: the Gaussian beam on a telescope's detector plane, drawn over the
  camera image (ported from LAGER ``BeamProjector.image_map``).

Frames: W world ENU about reference_point; C camera, OpenCV (x = image right, y = image down, z = LOS),
X_c = R_CW (X_w - C); H horn, R_CH = GaussianBeam.horn_to_camera(Ec), R_WH = R_CW^T R_CH.
"""

import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.transform import Rotation

# camera FRD in NED <-> OpenCV camera, as BOCK ekf.camera_ned_frd / IPA_flight AttitudeReconstruction
P_ENU2NED = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])
M_FRD2CAM = np.array([[0.0, 1, 0], [0, 0, 1], [1, 0, 0]])
UP = np.array([0.0, 0.0, 1.0])
ANG = ("yaw", "pitch", "roll")


def _wrap_pol_deg(a):
    """Polarization angle (period 180 deg) wrapped to [-90, 90)."""
    return (np.asarray(a, float) + 90.0) % 180.0 - 90.0


# ─────────────────────────────── BOCK EKF -> p5 ───────────────────────────────
def bock_ekf_to_p5(bock_version, p5, mount_rotation_zyx_deg=(0.0, 0.0, 0.0)) -> pl.DataFrame:
    """BOCK EKF states -> the format of IPA_flight ``dictionary_p5.ecsv``, one row per EKF state (IMU rate).

    Parameters
    ----------
    bock_version : pils.analyze.bock.BOCKVersion
        Its ``ekf_states`` table (t, kind, yaw/pitch/roll_cam_deg ± sig, dp_E/N/U) and ``ekf_photo_epochs``
        (frame, rms_px of the photo updates).
    p5 : pl.DataFrame | pd.DataFrame
        The photogrammetry p5 (frame, time, drone_E/N/U): camera positions and the time -> frame map.
    mount_rotation_zyx_deg : (3,)
        Camera mount correction of the attitude angles (photogrammetry config), the one BOCK's angles carry.

    Returns
    -------
    pl.DataFrame
        frame, time, quat_x/y/z/w_corr, rvec_x/y/z_corr, drone_E/N/U, projection_error, yaw[deg], pitch[deg],
        roll[deg] (as p5) + kind ("photo" / "imu"), sig_yaw[deg], sig_pitch[deg], sig_roll[deg].
        - rotation: R_CW rebuilt from the EKF camera angles (inverse of BOCK camera_ned_frd);
        - position: p5 drone_E/N/U linearly interpolated at the EKF time + the EKF camera-centre correction dp;
        - frame: camera frame interpolated at the EKF time (fractional between frames, the frame number on the
          photo updates), unique and increasing as p6 needs;
        - projection_error: EKF pixel rms on the photo updates, NaN elsewhere.
        Rows outside the p5 time range are dropped.
    """
    st = bock_version.tables["ekf_states"].sort_values("t").drop_duplicates("t", keep="first")
    p5 = p5.to_pandas() if hasattr(p5, "to_pandas") else pd.DataFrame(p5)
    p5 = p5.dropna(subset=["time", "drone_E", "drone_N", "drone_U"]).sort_values("time")
    t, tp = st["t"].to_numpy(float), p5["time"].to_numpy(float)
    st = st[(t >= tp[0]) & (t <= tp[-1])].reset_index(drop=True)
    t = st["t"].to_numpy(float)

    # R_CW from the EKF camera FRD angles in NED: C = P (M R_CW)^T F  =>  R_CW = M^T F C^T P
    ypr = np.array(st[[f"{a}_cam_deg" for a in ANG]], dtype=float)
    C = Rotation.from_euler("ZYX", ypr, degrees=True).as_matrix()
    M = Rotation.from_euler("ZYX", np.asarray(mount_rotation_zyx_deg, float).reshape(3), degrees=True).as_matrix()
    R_CW = M.T @ M_FRD2CAM @ np.transpose(C, (0, 2, 1)) @ P_ENU2NED
    rot = Rotation.from_matrix(R_CW)
    rvec, quat = rot.as_rotvec(), rot.as_quat()          # quat scalar-last, as IPA_flight rvec2quat

    pos = np.column_stack([np.interp(t, tp, p5[c].to_numpy(float)) for c in ("drone_E", "drone_N", "drone_U")])
    dp = np.array(st[["dp_E", "dp_N", "dp_U"]], dtype=float) if "dp_E" in st else np.zeros((len(st), 3))
    pos = pos + dp

    frame = np.interp(t, tp, p5["frame"].to_numpy(float))
    proj_err = np.full(len(st), np.nan)
    ep = bock_version.tables.get("ekf_photo_epochs")
    if ep is not None and {"t", "rms_px"} <= set(ep.columns):
        idx = np.searchsorted(t, ep["t"].to_numpy(float))
        ok = (idx < len(t)) & (np.abs(t[np.minimum(idx, len(t) - 1)] - ep["t"].to_numpy(float)) < 1e-9)
        proj_err[idx[ok]] = ep["rms_px"].to_numpy(float)[ok]

    out = {
        "frame": frame, "time": t,
        "quat_x_corr": quat[:, 0], "quat_y_corr": quat[:, 1], "quat_z_corr": quat[:, 2], "quat_w_corr": quat[:, 3],
        "rvec_x_corr": rvec[:, 0], "rvec_y_corr": rvec[:, 1], "rvec_z_corr": rvec[:, 2],
        "drone_E": pos[:, 0], "drone_N": pos[:, 1], "drone_U": pos[:, 2],
        "projection_error": proj_err,
        "yaw[deg]": ypr[:, 0], "pitch[deg]": ypr[:, 1], "roll[deg]": ypr[:, 2],
        "kind": st["kind"].astype(str).to_numpy(),
        **{f"sig_{a}[deg]": st[f"sig_{a}_cam_deg"].to_numpy(float) for a in ANG if f"sig_{a}_cam_deg" in st},
    }
    # one array per column, no pandas block: under pandas 3, pl.from_pandas zero-copies the float block and
    # multi-column .to_numpy() then returns a read-only view, which scipy's Rotation (p6) rejects
    return pl.DataFrame({k: np.array(v) for k, v in out.items()})


# ─────────────────────────────── EKF plot ───────────────────────────────
def plot_bock_ekf(bock_version, p5, outpath=None, title=None):
    """EKF camera yaw/pitch/roll in NED (±1σ band) vs the p5 photogrammetry angles, and their difference.
    Saved to outpath (returns it) if given, else returns the figure (notebook)."""
    import matplotlib.pyplot as plt

    st = bock_version.tables["ekf_states"].sort_values("t")
    p5 = p5.to_pandas() if hasattr(p5, "to_pandas") else pd.DataFrame(p5)
    p5 = p5.dropna(subset=["time"]).sort_values("time")
    t0 = st["t"].min()
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for ax, a in zip(axes, ANG):
        ekf = st[f"{a}_cam_deg"].to_numpy(float)
        ts = st["t"] - t0
        ax.plot(ts, ekf, lw=0.8, color="tab:blue", label="BOCK EKF")
        if f"sig_{a}_cam_deg" in st:
            s = st[f"sig_{a}_cam_deg"].to_numpy(float)
            ax.fill_between(ts, ekf - s, ekf + s, color="tab:blue", alpha=0.25, lw=0, label="EKF ±1σ")
        ax.plot(p5["time"] - t0, p5[f"{a}[deg]"], ".", ms=3, color="tab:orange", label="photogrammetry p5")
        ax.set_ylabel(f"{a} [deg]")
        ax.grid(True, alpha=0.3)
        ekf_unwrapped = np.degrees(np.unwrap(np.radians(ekf)))
        d = (p5[f"{a}[deg]"].to_numpy(float) - np.interp(p5["time"], st["t"], ekf_unwrapped) + 180.0) % 360.0 - 180.0
        axes[3].plot(p5["time"] - t0, d, ".", ms=3, label=f"{a}  (median {np.nanmedian(d):+.2f}°)")
    axes[0].legend(loc="best", fontsize=8, markerscale=3)
    axes[3].set_ylabel("p5 − EKF [deg]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(fontsize=8, markerscale=3, ncol=3)
    axes[3].set_xlabel("time from the first EKF state [s]")
    fig.suptitle(title or f"BOCK EKF camera attitude (NED) vs photogrammetry p5 -- {bock_version.version_name}")
    fig.tight_layout()
    if outpath is None:
        return fig
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


# ─────────────────────────────── beam in image space ───────────────────────────────
def telescope_basis(source, telescope):
    """Detector-plane basis of the telescope for source(s) (..., 3), world ENU (= the p6 line-of-sight basis):
    k = unit(X_tel - X_src), xi = unit(U x k) (increasing azimuth), eta = k x xi (increasing elevation).
    Returns k, xi, eta, distance [m]."""
    rel = np.asarray(telescope, float) - np.asarray(source, float)
    dist = np.linalg.norm(rel, axis=-1)
    k = rel / dist[..., None]
    xi = np.cross(UP, k)
    xi = xi / np.linalg.norm(xi, axis=-1)[..., None]
    return k, xi, np.cross(k, xi), dist


def _field_on_plane(P, src, R_WH, ex, ey, d0, beam):
    """Horn field at points P (..., 3): I_db (vs boresight at distance d0), psi_deg (xi -> eta), theta_deg."""
    r = P - src
    dist = np.linalg.norm(r, axis=-1)
    d_h = (r / dist[..., None]) @ R_WH                                   # R_WH^T r_hat
    theta = np.arctan2(np.hypot(d_h[..., 0], d_h[..., 1]), d_h[..., 2])
    phi = np.arctan2(d_h[..., 1], d_h[..., 0])
    E_W = beam.field(theta, phi) @ R_WH.T / dist[..., None]
    E_xi, E_eta = E_W @ ex, E_W @ ey
    return dict(I_db=10 * np.log10(np.maximum((E_xi ** 2 + E_eta ** 2) * d0 ** 2, 1e-30)),
                psi_deg=_wrap_pol_deg(np.degrees(np.arctan2(E_eta, E_xi))), theta_deg=np.degrees(theta))


def _project(points_w, R_CW, C, camera_matrix, dist_coeffs):
    """World points (..., 3) -> pixels (..., 2) and depth along the LOS (...)."""
    import cv2
    P = np.asarray(points_w, float)
    depth = (P - C) @ R_CW[2]
    uv, _ = cv2.projectPoints(P.reshape(-1, 3), cv2.Rodrigues(R_CW)[0], -R_CW @ C, camera_matrix, dist_coeffs)
    return uv.reshape(P.shape[:-1] + (2,)), depth


def beam_image_map(beam, R_CW, C, telescopes, tel_name, camera_matrix, dist_coeffs, image_size, Ec=0.0,
                   step_px=None):
    """The beam on the detector plane of `tel_name` mapped onto the camera image of one pose.

    Every pixel of a regular grid is undistorted, turned into a world ray with the pose (R_CW, C), intersected with
    the detector plane of the telescope (perpendicular to the LOS through the telescope); the horn field there gives
    I_db and psi. The polarization direction is brought back to the image (psi_img_deg, from image x towards y).

    beam : IPA_flight GaussianBeam;  telescopes : {name: ENU (3,)};  image_size : (width, height).
    Returns dict: u, v, I_db, psi_deg, psi_img_deg, theta_deg (NaN where the ray misses the plane), tel_px,
    boresight_px, image_size, tel_name, at_telescope (theta_deg, gain_db, psi_deg, dpsi_deg = psi - psi_proj).
    """
    import cv2
    K = np.asarray(camera_matrix, float).reshape(3, 3)
    D = np.asarray(dist_coeffs, float).ravel()
    C = np.asarray(C, float)
    R_WH = R_CW.T @ beam.horn_to_camera(Ec)
    W, H = image_size
    step = step_px or max(1, int(round(max(W, H) / 160)))
    u, v = np.arange(step / 2, W, step), np.arange(step / 2, H, step)
    U, V = np.meshgrid(u, v)

    tel = np.asarray(telescopes[tel_name], float)
    k, ex, ey, d0 = telescope_basis(C, tel)
    xy = cv2.undistortPoints(np.column_stack([U.ravel(), V.ravel()]).reshape(-1, 1, 2).astype(np.float64),
                             K, D).reshape(-1, 2)
    ray_w = np.column_stack([xy, np.ones(len(xy))]) @ R_CW                          # R_CW^T ray_c
    ray_w /= np.linalg.norm(ray_w, axis=1, keepdims=True)
    rk = ray_w @ k
    t = np.where(rk > 1e-9, ((tel - C) @ k) / np.where(rk > 1e-9, rk, 1.0), np.nan)
    P = (C + t[:, None] * ray_w).reshape(U.shape + (3,))
    f = _field_on_plane(P, C, R_WH, ex, ey, d0, beam)

    psi = np.radians(f["psi_deg"])
    Q = P + 1e-3 * d0 * (np.cos(psi)[..., None] * ex + np.sin(psi)[..., None] * ey)
    dq = _project(np.nan_to_num(Q), R_CW, C, K, D)[0] - _project(np.nan_to_num(P), R_CW, C, K, D)[0]
    bad = ~np.isfinite(t.reshape(U.shape))
    for key in ("I_db", "psi_deg", "theta_deg"):
        f[key] = np.where(bad, np.nan, f[key])
    psi_img = np.where(bad, np.nan, _wrap_pol_deg(np.degrees(np.arctan2(dq[..., 1], dq[..., 0]))))

    tel_px = {}
    for name, pos in telescopes.items():
        pix, dep = _project(np.asarray(pos, float), R_CW, C, K, D)
        tel_px[name] = (float(pix[0]), float(pix[1])) if dep > 0 else None
    bpix, _ = _project(C + 1e3 * R_WH[:, 2], R_CW, C, K, D)

    at = _field_on_plane(tel, C, R_WH, ex, ey, d0, beam)
    x_h = R_WH[:, 0]
    psi_proj = np.degrees(np.arctan2(x_h @ ey, x_h @ ex))
    at_tel = dict(theta_deg=float(at["theta_deg"]), gain_db=float(at["I_db"]), psi_deg=float(at["psi_deg"]),
                  dpsi_deg=float(_wrap_pol_deg(at["psi_deg"] - psi_proj)), distance_m=float(d0))
    return dict(u=u, v=v, I_db=f["I_db"], psi_deg=f["psi_deg"], psi_img_deg=psi_img, theta_deg=f["theta_deg"],
                tel_px=tel_px, boresight_px=(float(bpix[0]), float(bpix[1])), image_size=(W, H),
                tel_name=tel_name, at_telescope=at_tel)


def plot_beam_image_map(im, image=None, outpath=None, title=None, vmin=None, alpha=0.55):
    """Beam power [dB] over the camera frame (BGR array, optional), polarization ticks (psi_img), telescopes (o)
    and horn boresight (x) in pixels. Saved to outpath if given; returns the figure."""
    import matplotlib.pyplot as plt
    W, H = im["image_size"]
    if vmin is None:
        vmin = max(-30.0, np.floor(np.nanmin(im["I_db"])) if np.isfinite(im["I_db"]).any() else -30.0)
    fig, ax = plt.subplots(figsize=(11, 11 * H / W + 0.8))
    if image is not None:
        ax.imshow(image[..., ::-1] if image.ndim == 3 else image, cmap="gray", extent=[0, W, H, 0])
    du, dv = im["u"][1] - im["u"][0], im["v"][1] - im["v"][0]
    ext = [im["u"][0] - du / 2, im["u"][-1] + du / 2, im["v"][-1] + dv / 2, im["v"][0] - dv / 2]
    h = ax.imshow(np.ma.masked_invalid(im["I_db"]), extent=ext, vmin=vmin, vmax=0, cmap="magma",
                  alpha=alpha if image is not None else 1.0, interpolation="bilinear")
    fig.colorbar(h, ax=ax, label="horn power vs boresight [dB]", fraction=0.035)
    s = max(1, len(im["u"]) // 22)
    L = 0.6 * s * du
    ps = np.radians(im["psi_img_deg"][::s, ::s])
    Uu, Vv = np.meshgrid(im["u"][::s], im["v"][::s])
    ok = np.isfinite(ps)
    ax.quiver(Uu[ok], Vv[ok], L * np.cos(ps[ok]), L * np.sin(ps[ok]), angles="xy", scale_units="xy", scale=1,
              headwidth=0, headlength=0, headaxislength=0, pivot="middle", color="c", width=0.0025)
    for name, p in im["tel_px"].items():
        if p is not None and -0.2 * W < p[0] < 1.2 * W and -0.2 * H < p[1] < 1.2 * H:
            ax.plot(*p, "o", mfc="none", mec="lime" if name == im["tel_name"] else "w", ms=10, mew=1.5)
            ax.annotate(name, p, xytext=(6, -6), textcoords="offset points", color="w", fontsize=8)
    ax.plot(*im["boresight_px"], "x", color="w", ms=10, mew=2)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    a = im["at_telescope"]
    ax.set_title((title or f"Beam on the {im['tel_name']} detector plane, in image space")
                 + f"\nat {im['tel_name']}: θ = {a['theta_deg']:.2f}°, gain {a['gain_db']:.2f} dB, "
                 f"ψ = {a['psi_deg']:.3f}° (ξ→η), Δψ beam = {a['dpsi_deg']:+.4f}°, d = {a['distance_m']:.0f} m",
                 fontsize=10)
    ax.set_xlabel("u [px]")
    ax.set_ylabel("v [px]")
    if outpath is not None:
        fig.savefig(outpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
    return fig
