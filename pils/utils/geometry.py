"""Pointing geometry in local ENU coordinates (source antenna vs telescope)."""

import numpy as np
from numpy.typing import ArrayLike


def compute_poi(
    east: ArrayLike,
    north: ArrayLike,
    up: ArrayLike,
    yaw_deg: ArrayLike,
    pitch_deg: ArrayLike,
    roll_deg: ArrayLike,
) -> np.ndarray:
    """Point where the antenna line of sight hits the ground plane (U = 0).

    Conventions: the line of sight at zero attitude is (0, 1, 0) (forward =
    North); yaw is positive clockwise, pitch positive nose up, roll positive
    right wing down. Roll does not move a line of sight along the roll axis,
    so it is accepted for symmetry only.

    Returns
    -------
    np.ndarray
        (n, 3) ENU points; NaN where the line of sight does not reach the
        ground (pointing up or horizontal).
    """
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)
    up = np.asarray(up, dtype=float)
    yaw = np.radians(-np.asarray(yaw_deg, dtype=float))  # clockwise -> negate
    pitch = np.radians(np.asarray(pitch_deg, dtype=float))
    del roll_deg

    # Base LOS (0, 1, 0): pitch about E, then yaw about U
    d_e = -np.sin(yaw) * np.cos(pitch)
    d_n = np.cos(yaw) * np.cos(pitch)
    d_u = np.sin(pitch)

    with np.errstate(divide="ignore", invalid="ignore"):
        t = -up / d_u
    valid = (np.abs(d_u) > 1e-9) & (t >= 0)

    poi = np.full((len(east), 3), np.nan)
    poi[valid, 0] = east[valid] + t[valid] * d_e[valid]
    poi[valid, 1] = north[valid] + t[valid] * d_n[valid]
    poi[valid, 2] = 0.0
    return poi


def beam_2d_gauss(
    x: ArrayLike, y: ArrayLike, fwhm_h: float, fwhm_e: float
) -> np.ndarray:
    """Normalised 2D Gaussian beam on the (x, y) grid, peak 1 at (0, 0).

    ``x`` runs along the H plane and ``y`` along the E plane; FWHMs and grid
    share the same angular unit. The array is indexed ``[y, x]``.
    """
    sigma_h = fwhm_h / (2 * np.sqrt(np.log(2)))
    sigma_e = fwhm_e / (2 * np.sqrt(np.log(2)))
    grid_x, grid_y = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    beam = np.exp(-(grid_x**2 / sigma_h**2 + grid_y**2 / sigma_e**2))
    return beam / beam.max()


def normalize(v: ArrayLike) -> np.ndarray:
    """Unit vector."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def rot_x(angle: float) -> np.ndarray:
    """Rotation matrix about x by ``angle`` (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def build_frame(
    origin: ArrayLike,
    target: ArrayLike,
    roll_deg: float,
    up: ArrayLike = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Homogeneous transform of a frame at ``origin`` looking at ``target``.

    x points from origin to target, z is the projection of ``up``
    orthogonal to x, and the frame is rolled about x by ``roll_deg``.
    """
    origin = np.asarray(origin, dtype=float)
    up_axis = normalize(up)
    x_axis = normalize(np.asarray(target, dtype=float) - origin)
    if abs(np.dot(x_axis, up_axis)) > 0.99:  # degenerate: looking along up
        up_axis = np.array([0.0, 1.0, 0.0])
    z_axis = normalize(up_axis - np.dot(up_axis, x_axis) * x_axis)
    y_axis = np.cross(z_axis, x_axis)

    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x_axis, y_axis, z_axis)) @ rot_x(
        np.radians(roll_deg)
    )
    transform[:3, 3] = origin
    return transform


def world_to_local(transform: np.ndarray, point: ArrayLike) -> np.ndarray:
    """World coordinates -> coordinates in the frame of ``transform``."""
    return (np.linalg.inv(transform) @ np.append(np.asarray(point, float), 1.0))[:3]


def skew(v: ArrayLike) -> np.ndarray:
    """Skew-symmetric (cross product) matrix of a 3-vector."""
    x, y, z = np.asarray(v, dtype=float)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def axis_angle_to_rotation_matrix(axis: ArrayLike, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula."""
    k = skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def rotation_matrix_to_euler_zyx(rotation: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles (yaw, pitch, roll) in radians of a rotation matrix."""
    if abs(rotation[2, 0]) < 1:
        pitch = -np.arcsin(rotation[2, 0])
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:  # gimbal lock
        pitch = np.pi / 2 if rotation[2, 0] <= -1 else -np.pi / 2
        roll = 0.0
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    return float(yaw), float(pitch), float(roll)


def relative_yaw_pitch_roll(a: ArrayLike, b: ArrayLike) -> tuple[float, float, float]:
    """Yaw, pitch, roll (radians) of the smallest rotation taking a onto b."""
    a = normalize(a)
    b = normalize(b)
    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-8:  # parallel
        return 0.0, 0.0, 0.0
    angle = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    return rotation_matrix_to_euler_zyx(
        axis_angle_to_rotation_matrix(axis / axis_norm, angle)
    )
