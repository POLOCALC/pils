"""
A focal plane read from a detector table instead of the SO hardware model.

For experiments outside SO (CLASS), the per-detector pointing comes as a
whitespace table with one row per detector:

    det_uid  wafer  feed  type  az_off  el_off

`az_off` / `el_off` are the detector's offsets from the boresight in degrees,
on the sky at boresight rotation 0 -- so they are the focal-plane (xi, eta)
of `so3g.proj.quat.rotation_xieta` (+xi towards larger azimuth, +eta towards
higher elevation), and everything downstream (boresight rotation, a planner's
band cut and pointing solve, the crossings) treats them exactly like the SO wafers.

Each named `wafer` (a CLASS module) becomes one `Wafer`:

  * its pixels are the feeds -- the optical detectors of a feed (both
    polarisations) are averaged into one position, like an SO pixel;
  * its outline -- what a flight's band is cut from (GOSE's `wafer_sections`)
    -- is the module's full hexagon of feeds, and its centre is the middle feed
    of that hexagon (`hexagon_module`). A CLASS module is a close-packed hexagon
    of 37 feeds (3 rings around one), but a camera can have feeds without a
    pointing solution, often on the edge: the hull of the measured feeds alone
    is then lopsided, and so are the bands cut from it. The hexagon is fitted to
    the feeds that were measured -- their lattice, then an affine map from the
    lattice to the sky, which absorbs the optics' distortion -- and the missing
    feeds only fill in the outline, never a pixel. With `outline: hull`, or when
    the fit fails, the outline is the convex hull of the measured feeds.

Rows without a pointing solution (nan offsets), rows whose wafer is not named
(nan -- give it a name in the file and it becomes a wafer of its own), and
non-optical detectors (only `optical_types` count; CLASS's DD/DS darks sit on
feed 0 with no horn) are left out.

Requires: numpy, scipy.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy.spatial import ConvexHull

from pils.geometry.focal_plane import Wafer

DEFAULT_OPTICAL_TYPES = ("H", "V")
MODULE_RINGS = 3  # a CLASS module: 1 + 6 + 12 + 18 = 37 feeds
_HEX_DIRS = np.array(
    [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
)  # axial lattice steps, CCW
_COLUMNS = ("det_uid", "wafer", "feed", "type", "az_off", "el_off")
_MISSING = {"", "nan", "none", "null", "-"}


def read_detector_table(path: str) -> list[dict]:
    """Read the rows of a detector table as dicts (strings for the labels, floats for the offsets)."""
    rows: list[dict] = []
    header: list[str] | None = None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if header is None:
                header = parts
                missing = set(_COLUMNS) - set(header)
                if missing:
                    raise ValueError(
                        f"{path}: detector table lacks column(s) {', '.join(sorted(missing))}"
                    )
                continue
            if len(parts) != len(header):
                raise ValueError(
                    f"{path}: row {line!r} has {len(parts)} fields, header has {len(header)}"
                )
            r = dict(zip(header, parts, strict=False))
            rows.append(
                {
                    "det_uid": r["det_uid"],
                    "wafer": r["wafer"],
                    "feed": r["feed"],
                    "type": r["type"],
                    "az_off": float(r["az_off"]),
                    "el_off": float(r["el_off"]),
                }
            )
    return rows


def wafers_from_table(
    path: str,
    optical_types: Sequence[str] = DEFAULT_OPTICAL_TYPES,
    outline: str = "hexagon",
) -> tuple[Wafer, ...]:
    """Make one `Wafer` per named wafer of the table (sorted by name).

    Each with its pixels, centre and outline. See the module docstring for
    what is left out.
    """
    optical = {str(t) for t in optical_types}
    feeds: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in read_detector_table(path):
        if r["wafer"].lower() in _MISSING or r["type"] not in optical:
            continue
        if not (math.isfinite(r["az_off"]) and math.isfinite(r["el_off"])):
            continue
        feeds[r["wafer"]][r["feed"]].append((r["az_off"], r["el_off"]))
    if not feeds:
        raise ValueError(
            f"{path}: no optical detector ({'/'.join(sorted(optical))}) with a pointing and a named wafer"
        )

    return tuple(
        wafer_from_pixels(
            name,
            np.deg2rad(np.array([np.mean(p, axis=0) for p in feeds[name].values()])),
            where=path,
            outline=outline,
        )
        for name in sorted(feeds)
    )


def wafer_from_pixels(
    name: str,
    pixels: np.ndarray,
    where: str = "",
    outline: str = "hexagon",
    rings: int = MODULE_RINGS,
) -> Wafer:
    """Make a `Wafer` around its pixels ((n, 2) xi, eta in radians at boresight 0).

    Its outline and centre (the fitted module hexagon, or the hull of the
    pixels -- see the module docstring) and the pixels themselves.
    """
    pix = np.asarray(pixels, float)
    label = f"{where + ': ' if where else ''}wafer {name!r}"
    if len(pix) < 3:
        raise ValueError(f"{label} has {len(pix)} pixel(s), too few for an outline")
    if outline not in ("hexagon", "hull"):
        raise ValueError(
            f"{label}: outline must be `hexagon` or `hull`, not {outline!r}"
        )
    fit = hexagon_module(pix, rings) if outline == "hexagon" else None
    if fit is None:
        if outline == "hexagon":
            print(
                f"WARNING: {label}: its pixels do not fit a {3 * rings * (rings + 1) + 1}-feed hexagon; "
                f"outline = the hull of its {len(pix)} measured pixels"
            )
        centre, corners = pix.mean(axis=0), pix[ConvexHull(pix).vertices]
    else:
        centre, corners = fit
    rel = corners - centre
    return Wafer(
        name,
        float(centre[0]),
        float(centre[1]),
        outline=tuple((float(x), float(y)) for x, y in rel),
        pixels=pix,
    )


def hexagon_module(pix: np.ndarray, rings: int = MODULE_RINGS, tol: float = 0.25):
    """
    Fit the full hexagon of feeds a module's measured feeds belong to.

    The feeds lie on a triangular lattice, bent a little by the optics. Its
    pitch is the median nearest-neighbour distance and its orientation the mean
    of the neighbour directions modulo 60 deg. Lattice coordinates (i, j) are
    handed out from the feed nearest the middle outwards, one neighbour step at a
    time -- each step is read against the lattice directions locally, so the
    distortion across the module never adds up -- and the module is the hexagon
    of `rings` rings around the lattice point that holds them all (the one
    nearest their mean when several do). An affine map fitted from (i, j) to the
    measured positions places the whole hexagon on the sky. Returns (centre, the
    6 corners counter-clockwise), both (xi, eta) in the units of `pix`, or None
    when the feeds are too few, a step is off the lattice by more than `tol`
    pitch, two feeds land on one lattice point, or they do not fit one hexagon.
    """
    from collections import deque

    from scipy.spatial import cKDTree

    pix = np.asarray(pix, float)
    if len(pix) < 7:
        return None
    tree = cKDTree(pix)
    d, idx = tree.query(pix, k=2)
    pitch = float(np.median(d[:, 1]))
    vec = pix[idx[:, 1]] - pix
    theta = (
        np.angle(np.mean(np.exp(6j * np.arctan2(vec[:, 1], vec[:, 0])))) / 6
    )  # directions modulo 60 deg
    basis = pitch * np.array(
        [
            [np.cos(theta), np.cos(theta + np.pi / 3)],
            [np.sin(theta), np.sin(theta + np.pi / 3)],
        ]
    )
    steps = (basis @ _HEX_DIRS.T).T  # the six neighbour steps on the sky

    start = int(np.argmin(np.hypot(*(pix - pix.mean(axis=0)).T)))
    coords: dict[int, tuple[int, int]] = {start: (0, 0)}
    queue = deque([start])
    while queue:
        k = queue.popleft()
        for n in tree.query_ball_point(pix[k], 1.5 * pitch):
            if n == k:
                continue
            step = pix[n] - pix[k]
            miss = np.hypot(*(steps - step).T)
            best = int(np.argmin(miss))
            if miss[best] > tol * pitch:
                continue  # not a nearest neighbour (e.g. across a missing feed)
            c = tuple(int(x) for x in np.add(coords[k], _HEX_DIRS[best]))
            if n in coords:
                if coords[n] != c:
                    return None  # the lattice does not close up
                continue
            coords[n] = c
            queue.append(n)
    if len(coords) < len(pix) or len(set(coords.values())) < len(coords):
        return (
            None  # a feed not reached through its neighbours, or two feeds on one point
        )
    ij = np.array([coords[k] for k in range(len(pix))])

    def hexdist(a):  # lattice steps from the origin, axial coordinates
        return (
            np.abs(a[..., 0]) + np.abs(a[..., 1]) + np.abs(a[..., 0] + a[..., 1])
        ) // 2

    mean = ij.mean(axis=0)
    best = None
    for ci in range(int(mean[0]) - rings - 1, int(mean[0]) + rings + 2):
        for cj in range(int(mean[1]) - rings - 1, int(mean[1]) + rings + 2):
            if hexdist(ij - (ci, cj)).max() <= rings:
                key = float(np.hypot(*((ci, cj) - mean)))
                if best is None or key < best[0]:
                    best = (key, np.array([ci, cj]))
    if best is None:
        return None
    # lattice -> sky: an affine fit on the measured feeds (pitch, shear and rotation of the optics)
    design = np.column_stack([ij, np.ones(len(ij))])
    coef, *_ = np.linalg.lstsq(design, pix, rcond=None)
    to_sky = lambda a: np.column_stack([np.asarray(a, float), np.ones(len(a))]) @ coef  # noqa: E731
    c = best[1]
    return to_sky([c])[0], to_sky(c + rings * _HEX_DIRS)
