"""Running ``rnx2rtkp`` on a configuration and scoring its solution."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils.analyze.ppk_tuning.configs import (
    Candidate,
)

logger = logging.getLogger(__name__)


def rtklib_binary(name: str) -> str:
    """Path of an RTKLIB program: in ``$PILS_RTKLIB_BIN``, or on PATH.

    If it is in neither, the bare name is returned and running it fails with
    the usual "not found" error.
    """
    folder = os.environ.get("PILS_RTKLIB_BIN")
    if folder and (Path(folder) / name).is_file():
        return str(Path(folder) / name)
    return shutil.which(name) or name


# -- runs and scores ----------------------------------------------------


def run_rnx2rtkp(
    config: Path, rover_obs: Path, base_obs: Path, nav: Path, pos: Path
) -> Path:
    """Run ``rnx2rtkp``; returns the .pos file."""
    subprocess.run(
        [
            rtklib_binary("rnx2rtkp"),
            "-k",
            str(config),
            "-o",
            str(pos),
            str(rover_obs),
            str(base_obs),
            str(nav),
        ],
        check=True,
        capture_output=True,
    )
    return pos


def read_pos(pos: Path) -> pl.DataFrame:
    """Read a .pos file as a table (schema names, UTC timestamp)."""
    from pils.analyze.ppkdata.PPK.pos_analyzer import POSAnalyzer

    if not pos.exists() or pos.stat().st_size == 0:
        return pl.DataFrame()
    return POSAnalyzer(str(pos)).parse()


def compare_to_reference(
    solution: pl.DataFrame, reference: pl.DataFrame
) -> tuple[float, float, float] | None:
    """RMS (horizontal, vertical, m) and % of fixed epochs off by > 1 m.

    ``reference`` has ``timestamp``, ``lat_deg``, ``lon_deg``, ``h_ell_m``
    (e.g. the drone's RTK track); the median offset (lever arm) is removed.
    """
    import pymap3d as pm

    fixed = solution.filter(pl.col("quality") == 1).sort("timestamp")
    if fixed.height < 10 or reference.height < 10:
        return None
    reference = reference.sort("timestamp")
    origin = (
        float(reference["lat_deg"].median()),  # type: ignore[arg-type]
        float(reference["lon_deg"].median()),  # type: ignore[arg-type]
        float(reference["h_ell_m"].median()),  # type: ignore[arg-type]
    )
    t = fixed["timestamp"].to_numpy()
    rt = reference["timestamp"].to_numpy()
    inside = (t > rt[0]) & (t < rt[-1])
    if inside.sum() < 10:
        return None
    a = np.array(
        pm.geodetic2enu(
            fixed["lat_deg"].to_numpy(),
            fixed["lon_deg"].to_numpy(),
            fixed["h_ell_m"].to_numpy(),
            *origin,
        )
    )[:, inside]
    b = np.array(
        pm.geodetic2enu(
            reference["lat_deg"].to_numpy(),
            reference["lon_deg"].to_numpy(),
            reference["h_ell_m"].to_numpy(),
            *origin,
        )
    )
    b = np.array([np.interp(t[inside], rt, axis) for axis in b])
    residual = a - b
    residual -= np.median(residual, axis=1, keepdims=True)
    horizontal = np.hypot(residual[0], residual[1])
    vertical = np.abs(residual[2])
    return (
        float(np.sqrt(np.mean(horizontal**2))),
        float(np.sqrt(np.mean(vertical**2))),
        float(100 * np.mean((horizontal > 1.0) | (vertical > 1.0))),
    )


GOOD_H_M, GOOD_V_M = 0.10, 0.15
"""A fixed epoch is good within these (horizontal, vertical) of the reference."""

WRONG_M = 0.5
"""A fixed epoch is wrong (wrong ambiguities) when off by more than this."""


def score_against_reference(
    solution: pl.DataFrame, reference: pl.DataFrame
) -> dict[str, float] | None:
    """Score each fixed epoch against an independent track.

    The median offset of the fixed epochs (lever arm) is removed.

    Parameters
    ----------
    solution : pl.DataFrame
        ``timestamp``, ``lat_deg``, ``lon_deg``, ``h_ell_m``, ``quality``.
    reference : pl.DataFrame
        ``timestamp``, ``lat_deg``, ``lon_deg``, ``h_ell_m`` (e.g. the
        drone RTK track).

    Returns
    -------
    dict or None
        ``fix_pct``, ``good_pct`` (fixed within ``GOOD_H_M``/``GOOD_V_M``)
        and ``wrong_pct`` (fixed, off by more than ``WRONG_M``), all in % of
        the epochs inside the reference span; ``h_rms``/``v_rms`` (m) of
        the fixed epochs. None if the solution does not overlap the
        reference.
    """
    import pymap3d as pm

    reference = reference.sort("timestamp")
    rt = reference["timestamp"].to_numpy()
    sol = solution.filter(pl.col("timestamp").is_between(rt[0], rt[-1]))
    if sol.height == 0:
        return None
    fixed = sol.filter(pl.col("quality") == 1).sort("timestamp")
    out = {
        "fix_pct": 100 * fixed.height / sol.height,
        "good_pct": 0.0,
        "wrong_pct": 0.0,
        "h_rms": float("nan"),
        "v_rms": float("nan"),
    }
    if fixed.height < 10:
        return out
    origin = (
        float(reference["lat_deg"].median()),  # type: ignore[arg-type]
        float(reference["lon_deg"].median()),  # type: ignore[arg-type]
        float(reference["h_ell_m"].median()),  # type: ignore[arg-type]
    )
    a = np.array(
        pm.geodetic2enu(
            fixed["lat_deg"].to_numpy(),
            fixed["lon_deg"].to_numpy(),
            fixed["h_ell_m"].to_numpy(),
            *origin,
        )
    )
    b = np.array(
        pm.geodetic2enu(
            reference["lat_deg"].to_numpy(),
            reference["lon_deg"].to_numpy(),
            reference["h_ell_m"].to_numpy(),
            *origin,
        )
    )
    b = np.array([np.interp(fixed["timestamp"].to_numpy(), rt, axis) for axis in b])
    r = a - b
    r -= np.median(r, axis=1, keepdims=True)
    h, v = np.hypot(r[0], r[1]), np.abs(r[2])
    out["good_pct"] = 100 * float(np.sum((h < GOOD_H_M) & (v < GOOD_V_M))) / sol.height
    out["wrong_pct"] = 100 * float(np.sum((h > WRONG_M) | (v > WRONG_M))) / sol.height
    out["h_rms"] = float(np.sqrt(np.mean(h**2)))
    out["v_rms"] = float(np.sqrt(np.mean(v**2)))
    return out


def compare_directions(
    forward: pl.DataFrame, backward: pl.DataFrame
) -> tuple[float, float, float] | None:
    """Forward vs backward fixed solutions: RMS (h, v) and % disagreeing > 0.2 m."""
    import pymap3d as pm

    both = forward.filter(pl.col("quality") == 1).join(
        backward.filter(pl.col("quality") == 1), on="timestamp", suffix="_b"
    )
    if both.height < 10:
        return None
    e, n, u = pm.geodetic2enu(
        both["lat_deg_b"].to_numpy(),
        both["lon_deg_b"].to_numpy(),
        both["h_ell_m_b"].to_numpy(),
        both["lat_deg"].to_numpy(),
        both["lon_deg"].to_numpy(),
        both["h_ell_m"].to_numpy(),
    )
    horizontal = np.hypot(e, n)
    vertical = np.abs(u)
    return (
        float(np.sqrt(np.mean(horizontal**2))),
        float(np.sqrt(np.mean(vertical**2))),
        float(100 * np.mean((horizontal > 0.2) | (vertical > 0.2))),
    )


SETTING_CHOICES: dict[
    str, tuple[Callable[[dict[str, str]], bool], Callable[[dict[str, str]], bool]]
] = {
    "L1+L2 (vs L1)": (
        lambda o: o["pos1-frequency"] == "l1+l2",
        lambda o: o["pos1-frequency"] == "l1",
    ),
    "with Galileo": (
        lambda o: int(o["pos1-navsys"]) & 8 > 0,
        lambda o: int(o["pos1-navsys"]) & 8 == 0,
    ),
    "with GLONASS": (
        lambda o: int(o["pos1-navsys"]) & 4 > 0,
        lambda o: int(o["pos1-navsys"]) & 4 == 0,
    ),
    "with BeiDou": (
        lambda o: int(o["pos1-navsys"]) & 32 > 0,
        lambda o: int(o["pos1-navsys"]) & 32 == 0,
    ),
    "fix-and-hold (vs continuous)": (
        lambda o: o["pos2-armode"] == "fix-and-hold",
        lambda o: o["pos2-armode"] == "continuous",
    ),
    "SNR mask >= 33 dB-Hz (vs off or <= 27)": (
        lambda o: (
            o["pos1-snrmask_r"] == "on"
            and int(o["pos1-snrmask_L1"].split(",")[0]) >= 33
        ),
        lambda o: (
            o["pos1-snrmask_r"] == "off"
            or int(o["pos1-snrmask_L1"].split(",")[0]) <= 27
        ),
    ),
    "elevation mask >= 18 deg (vs <= 12)": (
        lambda o: int(o["pos1-elmask"]) >= 18,
        lambda o: int(o["pos1-elmask"]) <= 12,
    ),
    "AR threshold >= 3 (vs < 2.3)": (
        lambda o: float(o["pos2-arthres"]) >= 3.0,
        lambda o: float(o["pos2-arthres"]) < 2.3,
    ),
    "min fix sats >= 6 (vs <= 4)": (
        lambda o: int(o["pos2-minfixsats"]) >= 6,
        lambda o: int(o["pos2-minfixsats"]) <= 4,
    ),
}
"""Choices compared by :func:`setting_effects`: (with, without) on the overrides."""


def setting_effects(
    trials: list[Candidate], min_runs: int = 20
) -> list[dict[str, Any]]:
    """Measure how each setting acted on this data, from the random trials.

    The random configurations differ in all settings at once, so the runs
    with a choice and the runs without it differ, on average, only by that
    choice. For each choice of :data:`SETTING_CHOICES`:

    - ``gain``: the 97th percentile of good fixes (% of epochs; the fix
      rate when the trials were not scored against a reference) with the
      choice minus without, i.e. what the best configurations reach;
    - ``risk``: the mean wrong fixes (% of epochs) with minus without (None
      without a reference).

    On the November 2025 flights the effects depend on the data: e.g.
    fix-and-hold gained most with a weak Galileo E5b signal, GLONASS with
    few strong GPS satellites, BeiDou with GPS loss of lock; a ratio
    threshold of 3 and 6 satellites to fix cut wrong fixes by 3-5 % of
    epochs. A model of these effects learned on other days did not choose
    better than random sampling, so they are measured on each flight.

    Parameters
    ----------
    trials : list of Candidate
        Run candidates; only the ``random_*`` ones are used.
    min_runs : int, default=20
        Fewest runs on each side to report a choice.

    Returns
    -------
    list of dict
        ``choice``, ``gain``, ``risk``, ``runs_with``, ``runs_without``,
        sorted by gain.
    """
    runs = [t for t in trials if t.name.startswith("random_") and t.epochs]
    scored = bool(runs) and all(t.good_pct is not None for t in runs)

    def value(t: Candidate) -> float:
        return float(t.good_pct if scored and t.good_pct is not None else t.fix_pct)

    out = []
    for name, (yes, no) in SETTING_CHOICES.items():
        try:
            a = [t for t in runs if yes(t.overrides)]
            b = [t for t in runs if no(t.overrides)]
        except (KeyError, ValueError):
            continue
        if len(a) < min_runs or len(b) < min_runs:
            continue
        out.append(
            {
                "choice": name,
                "gain": float(
                    np.percentile([value(t) for t in a], 97)
                    - np.percentile([value(t) for t in b], 97)
                ),
                "risk": float(
                    np.mean([t.wrong_pct or 0.0 for t in a])
                    - np.mean([t.wrong_pct or 0.0 for t in b])
                )
                if scored
                else None,
                "runs_with": len(a),
                "runs_without": len(b),
            }
        )
    return sorted(out, key=lambda e: -e["gain"])
