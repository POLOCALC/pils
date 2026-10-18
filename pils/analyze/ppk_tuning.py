"""Tune the RTKLIB configuration to the data of a flight.

The best RTKLIB settings depend on the data: which signals the rover tracks
well (a payload antenna may have a weak L2), which constellations both
receivers see, how fast the data are. :func:`tune` profiles the rover and base
RINEX files, builds candidate configurations around a template (a grid from
the profile and :data:`N_RANDOM` random configurations), runs them all with
``rnx2rtkp`` in parallel, and keeps the best one that passes validation:

- with a reference track (the drone's RTK solution, flown about a metre from
  the payload antenna), every fixed epoch is scored against it after
  removing the median offset (lever arm): the most good fixes among the
  candidates with at most ``max_wrong_pct`` % wrong fixes;
- without one, the forward and backward solutions must agree on the epochs
  that both fix; this check is weaker (both directions can hold the same
  wrong ambiguities with fix-and-hold, which is then not tried).

The fix rate alone is not a safe criterion: lower ratio thresholds and
fix-and-hold raise it with wrong fixes (on flight_20251206_1833, 72 % fixed
but 1.5 % of the fixes off by more than a metre, against 65 % with none).

The random configurations also show how each setting acted on the flight
(:func:`setting_effects`).

Examples
--------
>>> result = tune(template, rover_obs, base_obs, nav, workdir,
...               reference=drone_track)  # doctest: +SKIP
>>> result.best.overrides, result.best.good_pct  # doctest: +SKIP
>>> result.effects  # doctest: +SKIP
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# RTKLIB navigation system bits and RINEX system letters
NAVSYS_BITS = {"G": 1, "S": 2, "R": 4, "E": 8, "J": 16, "C": 32, "I": 64}
# Rover data that limits fixing whatever the configuration: L1 phase loss of
# lock flagged on more than this fraction of observations, or continuous
# tracking shorter than this (s). flight_20251206_1833 (67 % fixed): 4 %,
# 388 s; flight_20250428_1625 (no fix): 15 %, 95 s
MAX_ROVER_LLI = 0.10
MIN_ROVER_ARC_S = 150.0


def data_quality_note(rover: ObsProfile) -> str:
    """Why the rover data may limit fixing, or "" if it looks fine."""
    issues = []
    if rover.lli_fraction > MAX_ROVER_LLI:
        issues.append(f"loss of lock on {100 * rover.lli_fraction:.0f} % of L1 phase")
    if 0 < rover.arc_s < MIN_ROVER_ARC_S:
        issues.append(f"continuous tracking only ~{rover.arc_s:.0f} s")
    if not issues:
        return ""
    return "rover data limits fixing: " + ", ".join(issues)


# Minimum median number of satellites a system must have in common between
# rover and base to be used
MIN_COMMON_SATS = 4
# Rover L2 is used only if its median SNR reaches this (dB-Hz)
MIN_L2_SNR = 30.0


def rtklib_binary(name: str) -> str:
    """Path of an RTKLIB program: in ``$PILS_RTKLIB_BIN``, or on PATH.

    If it is in neither, the bare name is returned and running it fails with
    the usual "not found" error.
    """
    folder = os.environ.get("PILS_RTKLIB_BIN")
    if folder and (Path(folder) / name).is_file():
        return str(Path(folder) / name)
    return shutil.which(name) or name


# -- data profile -------------------------------------------------------


@dataclass
class ObsProfile:
    """What a RINEX 3 observation file contains."""

    interval_s: float
    start: np.datetime64 | None
    end: np.datetime64 | None
    epochs: int
    # Median number of satellites per epoch, by system letter
    sats: dict[str, float] = field(default_factory=dict)
    # Median SNR (dB-Hz) by system and band ("1", "2", "5", "7", ...)
    snr: dict[str, dict[str, float]] = field(default_factory=dict)
    # Fraction of satellite-epochs with the band observed
    band_fraction: dict[str, dict[str, float]] = field(default_factory=dict)
    # APPROX POSITION XYZ of the header (receiver's own solution), m
    approx_xyz: tuple[float, float, float] | None = None
    # Fraction of L1 phase observations flagged loss of lock (LLI bit 0)
    lli_fraction: float = 0.0
    # Continuous L1 phase tracking per observation: mean length of the arc an
    # observation belongs to (sum of squared arc lengths / total), s
    arc_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the profile as a dict, for the metadata."""
        return {
            "interval_s": self.interval_s,
            "start": str(self.start),
            "end": str(self.end),
            "epochs": self.epochs,
            "sats": self.sats,
            "snr": self.snr,
            "band_fraction": self.band_fraction,
            "approx_xyz": self.approx_xyz,
            "lli_fraction": self.lli_fraction,
            "arc_s": self.arc_s,
        }


def profile_obs(path: str | Path, max_epochs: int = 3000) -> ObsProfile:
    """Profile a RINEX 3 observation file (SNR, satellites, interval).

    Up to ``max_epochs`` epochs, evenly spread over the file, are read for
    the SNR and satellite statistics; the whole file for times and count.
    """
    path = Path(path)
    types: dict[str, list[str]] = {}
    approx = None
    with open(path, errors="replace") as fh:
        for line in fh:
            label = line[60:].strip()
            if label == "APPROX POSITION XYZ":
                xyz = tuple(float(v) for v in line[:60].split()[:3])
                approx = xyz if any(xyz) else None
            if label == "SYS / # / OBS TYPES":
                if line[0] != " ":
                    system = line[0]
                    types[system] = []
                types[system] += line[7:58].split()
            if label == "END OF HEADER":
                break
        body = fh.readlines()

    epoch_rows = [i for i, line in enumerate(body) if line.startswith(">")]
    times = [_epoch_time(body[i]) for i in epoch_rows]
    step = max(1, len(epoch_rows) // max_epochs)
    snr: dict[str, dict[str, list[float]]] = {}
    present: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    sats: dict[str, list[int]] = {}
    for k in range(0, len(epoch_rows), step):
        start = epoch_rows[k] + 1
        stop = epoch_rows[k + 1] if k + 1 < len(epoch_rows) else len(body)
        count: dict[str, int] = {}
        for line in body[start:stop]:
            system = line[0]
            if system not in types:
                continue
            count[system] = count.get(system, 0) + 1
            totals[system] = totals.get(system, 0) + 1
            for j, obs in enumerate(types[system]):
                if obs[0] != "S":
                    continue
                field_ = line[3 + 16 * j : 3 + 16 * j + 14].strip()
                if not field_:
                    continue
                band = obs[1]
                value = float(field_)
                if value <= 0:
                    continue
                snr.setdefault(system, {}).setdefault(band, []).append(value)
                present.setdefault(system, {})[band] = (
                    present.setdefault(system, {}).get(band, 0) + 1
                )
        for system in types:
            sats.setdefault(system, []).append(count.get(system, 0))

    # Loss of lock and continuous tracking of the L1 phase, over all epochs
    l1_col = {
        s: next((j for j, o in enumerate(t) if o.startswith("L1")), None)
        for s, t in types.items()
    }
    phase = flagged = 0
    current: dict[str, int] = {}
    arcs: list[int] = []
    for k, row in enumerate(epoch_rows):
        stop = epoch_rows[k + 1] if k + 1 < len(epoch_rows) else len(body)
        seen = set()
        for line in body[row + 1 : stop]:
            column = l1_col.get(line[0])
            if column is None:
                continue
            field_ = line[3 + 16 * column : 3 + 16 * column + 16]
            if not field_[:14].strip():
                continue
            phase += 1
            if field_[14:15].strip().isdigit() and int(field_[14:15]) & 1:
                flagged += 1
            seen.add(line[:3])
        for sat in list(current):
            if sat not in seen:
                arcs.append(current.pop(sat))
        for sat in seen:
            current[sat] = current.get(sat, 0) + 1
    arcs += current.values()

    deltas = np.diff(np.array(times, dtype="datetime64[ms]")).astype(float) / 1e3
    return ObsProfile(
        interval_s=float(np.median(deltas)) if deltas.size else float("nan"),
        start=np.datetime64(times[0], "ms") if times else None,
        end=np.datetime64(times[-1], "ms") if times else None,
        epochs=len(epoch_rows),
        sats={s: float(np.median(v)) for s, v in sats.items() if v},
        snr={
            s: {b: float(np.median(v)) for b, v in bands.items()}
            for s, bands in snr.items()
        },
        band_fraction={
            s: {b: n / totals[s] for b, n in bands.items()}
            for s, bands in present.items()
            if totals.get(s)
        },
        approx_xyz=approx,  # type: ignore[arg-type]
        lli_fraction=flagged / phase if phase else 0.0,
        arc_s=float(np.sum(np.square(arcs)) / np.sum(arcs)) * float(np.median(deltas))
        if arcs and deltas.size
        else 0.0,
    )


def _epoch_time(line: str) -> str:
    year, month, day, hour, minute = (int(x) for x in line[2:29].split()[:5])
    second = float(line[2:29].split()[5])
    whole = int(second)
    return (
        f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{whole:02d}."
        f"{round((second - whole) * 1000):03d}"
    )


# -- candidates ----------------------------------------------------------


@dataclass
class Candidate:
    """A configuration: the template with these settings changed."""

    name: str
    overrides: dict[str, str]
    fix_pct: float = 0.0
    epochs: int = 0
    h_rms: float | None = None
    v_rms: float | None = None
    bad_pct: float | None = None
    good_pct: float | None = None
    wrong_pct: float | None = None
    valid: bool = False
    note: str = ""

    def row(self) -> dict[str, Any]:
        """Return the candidate as a row of the trials table."""
        return {
            "name": self.name,
            **{f"set_{k}": v for k, v in self.overrides.items()},
            "epochs": self.epochs,
            "fix_pct": self.fix_pct,
            "h_rms_m": self.h_rms,
            "v_rms_m": self.v_rms,
            "bad_pct": self.bad_pct,
            "good_pct": self.good_pct,
            "wrong_pct": self.wrong_pct,
            "valid": self.valid,
            "note": self.note,
        }


def snr_masks(rover: ObsProfile, template: str = "") -> list[tuple[str, int | None]]:
    """List the SNR masks to try.

    The template's (if on), one from the rover's L1 SNR (median of GPS and
    Galileo minus 5 dB-Hz, within 20-35), and none.

    Returns (label, mask in dB-Hz or None for off).
    """
    masks: list[tuple[str, int | None]] = []
    if config_value(template, "pos1-snrmask_r") == "on":
        first = (config_value(template, "pos1-snrmask_L1") or "").split(",")[0]
        if first.strip().isdigit():
            masks.append((f"snr{int(first)}", int(first)))
    l1 = [v for s in ("G", "E") if (v := rover.snr.get(s, {}).get("1")) is not None]
    if l1:
        value = int(np.clip(round(float(np.median(l1))) - 5, 20, 35))
        if value not in [m for _, m in masks]:
            masks.append((f"snr{value}", value))
    masks.append(("snroff", None))
    return masks


def candidates(
    rover: ObsProfile, base: ObsProfile, hold: bool = True, template: str = ""
) -> list[Candidate]:
    """Candidate configurations for the data.

    - frequency: L1, and L1+L2 only if the rover tracks L2 well (median SNR
      >= ``MIN_L2_SNR`` and at least half of the satellite-epochs);
    - systems: GPS, plus every combination of Galileo, GLONASS and BeiDou
      that both receivers see enough of (``MIN_COMMON_SATS``);
    - rover SNR mask: the template's, one from the rover's L1 SNR, and off
      (:func:`snr_masks`);
    - elevation mask 10/15/20 deg; ambiguity resolution continuous or (with
      ``hold``) fix-and-hold; ratio threshold 2 or 3.
    """
    common = {
        s: min(rover.sats.get(s, 0), base.sats.get(s, 0)) for s in ("G", "E", "R", "C")
    }
    usable = [s for s in ("E", "R", "C") if common[s] >= MIN_COMMON_SATS]
    systems: list[list[str]] = []
    if common["G"] >= MIN_COMMON_SATS:
        systems.append(["G"])
        for k in range(1, len(usable) + 1):
            systems += [["G", *c] for c in itertools.combinations(usable, k)]

    frequencies = ["l1"]
    l2 = [
        rover.snr.get(s, {}).get("2", 0.0)
        for s in ("G", "R")
        if rover.band_fraction.get(s, {}).get("2", 0.0) >= 0.5
    ]
    if l2 and np.median(l2) >= MIN_L2_SNR:
        frequencies.append("l1+l2")

    out = []
    for freq, navsys, (snr_label, snr), el, armode, thres in itertools.product(
        frequencies,
        systems,
        snr_masks(rover, template),
        (10, 15, 20),
        ("continuous", "fix-and-hold") if hold else ("continuous",),
        (2, 3),
    ):
        bits = sum(NAVSYS_BITS[s] for s in navsys)
        overrides = {
            "pos1-frequency": freq,
            "pos1-navsys": str(bits),
            "pos1-elmask": str(el),
            "pos2-armode": armode,
            "pos2-arthres": str(thres),
            "pos2-arthresmin": str(thres),
            "pos2-arthresmax": str(thres),
            "pos1-soltype": "combined",
            "pos1-snrmask_r": "off" if snr is None else "on",
        }
        if snr is not None:
            overrides["pos1-snrmask_L1"] = ",".join([str(snr)] * 9)
            overrides["pos1-snrmask_L2"] = ",".join([str(snr)] * 9)
        out.append(
            Candidate(
                name=f"{freq}_{''.join(navsys)}_{snr_label}_el{el}_{armode}_th{thres}",
                overrides=overrides,
            )
        )
    return out


# The template's base position is not used if it is farther than this from
# the base receiver's own position (RINEX header), m
N_RANDOM = 1000
"""Random configurations tried by :func:`tune` besides the grid.

On the 119 November 2025 flights, the best of 1000 random configurations
(validated on the drone RTK track, at most 0.5 % wrong fixes) gave a median
of 9 % correctly fixed epochs, twice the grid's; 1000 runs take about a
minute on 32 cores.
"""


def usable_systems(rover: ObsProfile, base: ObsProfile) -> list[str]:
    """List GPS and the systems both receivers see enough of."""
    common = {
        s: min(rover.sats.get(s, 0), base.sats.get(s, 0)) for s in ("G", "E", "R", "C")
    }
    return [s for s in ("G", "E", "R", "C") if common[s] >= MIN_COMMON_SATS]


def random_candidates(
    rover: ObsProfile,
    base: ObsProfile,
    n: int = N_RANDOM,
    seed: int = 0,
    hold: bool = True,
) -> list[Candidate]:
    """Draw random configurations over 18 RTKLIB settings.

    The settings and their ranges: frequency (L1, L1+L2), systems (GPS plus
    any of the usable Galileo, GLONASS, BeiDou), elevation mask 5-25 deg,
    rover SNR mask off or 20-40 dB-Hz, ambiguity resolution continuous or
    (with ``hold``) fix-and-hold, ratio threshold 1.5-4, GLONASS and BeiDou
    ambiguity resolution, lock count 0-20, minimum satellites to fix 3-8,
    ambiguity elevation mask 0-25 deg, outage count 1-20, slip threshold
    0.02-0.2 m, ionosphere rejection 1-30 m, code/phase error ratio
    100-500, phase error 2-20 mm, horizontal acceleration noise 1-10 m/s2,
    receiver dynamics.

    Parameters
    ----------
    rover, base : ObsProfile
        Data profiles (:func:`profile_obs`); they decide the systems.
    n : int, default=N_RANDOM
        Number of configurations.
    seed : int, default=0
        Random seed (the same data and seed give the same candidates).
    hold : bool, default=True
        Allow fix-and-hold.

    Returns
    -------
    list of Candidate
        Named ``random_0000`` ... with their overrides.
    """
    rng = np.random.default_rng(seed)
    others = [s for s in usable_systems(rover, base) if s != "G"]
    out = []
    for i in range(n):
        systems = ["G"] + [s for s in others if rng.random() < 0.5]
        snr = int(rng.choice([0, *range(20, 41)]))
        thres = float(np.round(rng.uniform(1.5, 4.0), 2))
        phase = float(np.round(10 ** rng.uniform(np.log10(0.002), np.log10(0.02)), 4))
        ratio = str(int(rng.integers(100, 501)))
        mask = ",".join([str(snr or 30)] * 9)
        overrides = {
            "pos1-frequency": str(rng.choice(["l1", "l1+l2"])),
            "pos1-navsys": str(sum(NAVSYS_BITS[s] for s in systems)),
            "pos1-elmask": str(int(rng.integers(5, 26))),
            "pos1-snrmask_r": "on" if snr else "off",
            "pos1-snrmask_b": "off",
            "pos1-snrmask_L1": mask,
            "pos1-snrmask_L2": mask,
            "pos1-soltype": "combined",
            "pos1-dynamics": str(rng.choice(["on", "off"])),
            "pos2-armode": str(rng.choice(["continuous", "fix-and-hold"]))
            if hold
            else "continuous",
            "pos2-arthres": str(thres),
            "pos2-arthresmin": str(thres),
            "pos2-arthresmax": str(thres),
            "pos2-gloarmode": str(rng.choice(["off", "autocal"])),
            "pos2-bdsarmode": str(rng.choice(["off", "on"])),
            "pos2-arlockcnt": str(int(rng.integers(0, 21))),
            "pos2-minfixsats": str(int(rng.integers(3, 9))),
            "pos2-arelmask": str(int(rng.integers(0, 26))),
            "pos2-aroutcnt": str(int(rng.integers(1, 21))),
            "pos2-slipthres": str(
                float(np.round(10 ** rng.uniform(np.log10(0.02), np.log10(0.2)), 3))
            ),
            "pos2-rejionno": str(
                float(np.round(10 ** rng.uniform(0, np.log10(30)), 1))
            ),
            "stats-eratio1": ratio,
            "stats-eratio2": ratio,
            "stats-errphase": str(phase),
            "stats-errphaseel": str(phase),
            "stats-prnaccelh": str(float(np.round(rng.uniform(1, 10), 1))),
        }
        out.append(Candidate(name=f"random_{i:04d}", overrides=overrides))
    return out


MAX_BASE_OFFSET_M = 50.0


def config_value(text: str, key: str) -> str | None:
    """Return a setting of an RTKLIB config text (without its comment)."""
    match = re.search(rf"^{re.escape(key)}\s*=([^#\n]*)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def base_position_check(text: str, base: ObsProfile) -> tuple[dict[str, str], str]:
    """Check the template's base position against the base data.

    Returns the overrides (``ant2-postype=single``: the average of the base's
    own solutions) and a note, or ``({}, note)`` if the position is fine or
    cannot be checked.
    """
    import pymap3d as pm

    if base.approx_xyz is None:
        return {}, "base position not checked (no position in the base RINEX)"
    postype = config_value(text, "ant2-postype")
    try:
        p1, p2, p3 = (
            float(config_value(text, f"ant2-pos{i}") or "nan") for i in (1, 2, 3)
        )
    except ValueError:
        return {}, "base position not checked (template position unreadable)"
    if postype == "llh":
        xyz = pm.geodetic2ecef(p1, p2, p3)
    elif postype == "xyz":
        xyz = (p1, p2, p3)
    else:
        return {}, f"base position from the data (ant2-postype={postype})"
    offset = float(np.linalg.norm(np.subtract(xyz, base.approx_xyz)))
    if offset <= MAX_BASE_OFFSET_M:
        return {}, f"template base position {offset:.1f} m from the base receiver's"
    return (
        {"ant2-postype": "single"},
        f"template base position {offset:.0f} m from the base receiver's own: "
        "using the average of the base's single solutions (absolute position to "
        "a few metres; set the base coordinates in the template)",
    )


def apply_overrides(template: str, overrides: dict[str, str]) -> str:
    """Return the template config text with some settings replaced (or added)."""
    lines = template.splitlines()
    done = set()
    for i, line in enumerate(lines):
        match = re.match(r"^([\w-]+)\s*=", line)
        if match and match.group(1) in overrides:
            key = match.group(1)
            comment = line[line.index("#") :] if "#" in line else ""
            lines[i] = f"{key:<18} ={overrides[key]}" + (
                f"  {comment}" if comment else ""
            )
            done.add(key)
    lines += [f"{k:<18} ={v}" for k, v in overrides.items() if k not in done]
    return "\n".join(lines) + "\n"


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


@dataclass
class TuningResult:
    """Outcome of :func:`tune`."""

    best: Candidate | None
    trials: list[Candidate]
    rover: ObsProfile
    base: ObsProfile
    config_text: str | None
    validation: str
    base_note: str = ""
    data_note: str = ""

    @property
    def effects(self) -> list[dict[str, Any]]:
        """Effect of each setting on this data (see :func:`setting_effects`)."""
        return setting_effects(self.trials)

    def table(self) -> pl.DataFrame:
        """Return the trials as a table, one row per candidate."""
        return pl.DataFrame([c.row() for c in self.trials])

    def metadata(self) -> dict[str, Any]:
        """Return the tuning summary for the PPK revision's manifest."""
        return {
            "tuning_validation": self.validation,
            "tuning_base_position": self.base_note,
            "tuning_data_quality": self.data_note,
            "tuning_best": self.best.row() if self.best else None,
            "tuning_trials": len(self.trials),
            "tuning_effects": self.effects,
            "rover_profile": self.rover.as_dict(),
            "base_profile": self.base.as_dict(),
        }


def tune(
    template: str | Path,
    rover_obs: str | Path,
    base_obs: str | Path,
    nav: str | Path,
    workdir: str | Path,
    reference: pl.DataFrame | None = None,
    max_h_rms: float = 0.10,
    max_v_rms: float = 0.15,
    max_bad_pct: float = 0.5,
    workers: int | None = None,
    n_random: int = N_RANDOM,
    seed: int = 0,
    max_wrong_pct: float = 0.5,
    selection: str = "fix_rate",
) -> TuningResult:
    """Choose the RTKLIB configuration for this data.

    Parameters
    ----------
    template : str or Path
        RTKLIB configuration the candidates start from (base position,
        statistics, output format, ... come from it).
    rover_obs, base_obs, nav : str or Path
        RINEX observation files and navigation file; the base must cover
        the rover's time span.
    workdir : str or Path
        Folder for the candidates' configs and solutions.
    reference : pl.DataFrame, optional
        Independent track of the rover (``timestamp`` UNIX s UTC,
        ``lat_deg``, ``lon_deg``, ``h_ell_m``), e.g. the drone RTK track;
        used by ``selection="reference"``.
    max_h_rms, max_v_rms : float
        Largest RMS (m) of the fixed epochs against the reference.
    max_bad_pct : float
        Largest % of fixed epochs off by more than 1 m (0.2 m between
        directions without a reference).
    workers : int, optional
        Parallel runs (default: CPU count).
    n_random : int, default=N_RANDOM
        Random configurations tried besides the grid
        (:func:`random_candidates`); 0 for the grid only.
    seed : int, default=0
        Seed of the random configurations.
    max_wrong_pct : float, default=0.5
        With a reference: largest % of epochs with a wrong fix (off by more
        than ``WRONG_M``) for a valid candidate.
    selection : {"fix_rate", "reference", "directions"}, default="fix_rate"
        How the best candidate is chosen (see Notes).

    Returns
    -------
    TuningResult
        The best candidate (None if none is valid), every trial, the data
        profiles, the chosen config and the effect of each setting on this
        data (:func:`setting_effects`).

    Notes
    -----
    - ``"fix_rate"``: the candidate with the most epochs fixed by RTKLIB
      (Q=1, of all the solution's epochs); no other check. On the November
      2025 flights the highest fix rates often included wrong fixes (the
      template: 17 % of epochs fixed off by more than 0.5 m, on average).
    - ``"reference"``: every epoch is scored against ``reference``
      (:func:`score_against_reference`): valid with at most
      ``max_wrong_pct`` wrong fixes, the best has the most good fixes.
    - ``"directions"``: forward and backward solutions must agree (two runs
      per candidate, no fix-and-hold); the most fixes.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    text = Path(template).read_text() if Path(str(template)).exists() else str(template)
    rover, base = profile_obs(rover_obs), profile_obs(base_obs)
    base_overrides, base_note = base_position_check(text, base)
    data_note = data_quality_note(rover)
    if data_note:
        logger.warning(data_note)
    if base_overrides:
        logger.warning(base_note)
    if selection not in ("fix_rate", "reference", "directions"):
        raise ValueError(f"Unknown selection {selection!r}")
    if selection == "reference" and reference is None:
        raise ValueError('selection="reference" needs a reference track')
    directions = selection == "directions"
    by_reference = selection == "reference"
    # Forward and backward fix-and-hold solutions can hold the same wrong
    # ambiguities: with "directions", only continuous resolution is tried
    trials = candidates(rover, base, hold=not directions, template=text)
    if n_random:
        trials += random_candidates(
            rover, base, n=n_random, seed=seed, hold=not directions
        )
    method = {
        "fix_rate": "fix rate",
        "reference": "reference track",
        "directions": "forward/backward",
    }[selection]
    logger.info(f"Tuning RTKLIB: {len(trials)} candidates, selection by {method}")

    def run(candidate: Candidate) -> Candidate:
        candidate.overrides.update(base_overrides)
        overrides = dict(candidate.overrides)
        try:
            if directions:
                results = []
                for soltype in ("forward", "backward"):
                    overrides["pos1-soltype"] = soltype
                    conf = workdir / f"{candidate.name}_{soltype}.conf"
                    conf.write_text(apply_overrides(text, overrides))
                    results.append(
                        read_pos(
                            run_rnx2rtkp(
                                conf,
                                Path(rover_obs),
                                Path(base_obs),
                                Path(nav),
                                workdir / f"{candidate.name}_{soltype}.pos",
                            )
                        )
                    )
                solution = results[0]
                check = (
                    compare_directions(*results)
                    if all(r.height for r in results)
                    else None
                )
            else:
                conf = workdir / f"{candidate.name}.conf"
                conf.write_text(apply_overrides(text, overrides))
                solution = read_pos(
                    run_rnx2rtkp(
                        conf,
                        Path(rover_obs),
                        Path(base_obs),
                        Path(nav),
                        workdir / f"{candidate.name}.pos",
                    )
                )
                if not by_reference:
                    candidate.epochs = solution.height
                    if solution.height:
                        candidate.fix_pct = float(
                            100 * (solution["quality"] == 1).sum() / solution.height
                        )
                    candidate.valid = candidate.fix_pct > 0
                    if not candidate.valid:
                        candidate.note = "no fixed epochs"
                    return candidate
                assert reference is not None
                score = (
                    score_against_reference(solution, reference)
                    if solution.height
                    else None
                )
                candidate.epochs = solution.height
                if score is None:
                    candidate.note = "no epoch inside the reference track"
                    return candidate
                candidate.fix_pct = score["fix_pct"]
                candidate.good_pct = score["good_pct"]
                candidate.wrong_pct = score["wrong_pct"]
                candidate.h_rms, candidate.v_rms = score["h_rms"], score["v_rms"]
                candidate.valid = (
                    candidate.good_pct > 0 and candidate.wrong_pct <= max_wrong_pct
                )
                if not candidate.valid:
                    candidate.note = (
                        "no good fixes"
                        if candidate.good_pct == 0
                        else f"{candidate.wrong_pct:.1f} % wrong fixes"
                    )
                return candidate
        except subprocess.CalledProcessError as e:
            candidate.note = f"rnx2rtkp failed: {e.stderr[-200:]!r}"
            return candidate
        candidate.epochs = solution.height
        if solution.height:
            candidate.fix_pct = float(
                100 * (solution["quality"] == 1).sum() / solution.height
            )
        if check is not None:
            candidate.h_rms, candidate.v_rms, candidate.bad_pct = check
            candidate.valid = (
                candidate.h_rms <= max_h_rms
                and candidate.v_rms <= max_v_rms
                and candidate.bad_pct <= max_bad_pct
            )
            if not candidate.valid:
                candidate.note = "fixes do not pass validation"
        else:
            candidate.note = "too few fixed epochs to validate"
        return candidate

    with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        trials = list(pool.map(run, trials))

    valid = [c for c in trials if c.valid]

    def merit(c: Candidate) -> tuple[float, float]:
        if directions:
            return round(c.fix_pct, 1), -(c.h_rms or 0)
        if by_reference:
            return round(c.good_pct or 0.0, 1), -(c.wrong_pct or 0.0)
        return round(c.fix_pct, 1), 0.0

    best = max(valid, key=merit) if valid else None
    config_text = None
    if best is not None:
        config_text = apply_overrides(text, best.overrides)
        logger.info(
            f"Best RTKLIB config {best.name}: {best.fix_pct:.1f} % fixed"
            + (
                f", {best.good_pct:.1f} % good, {best.wrong_pct:.2f} % wrong"
                if by_reference
                else ""
            )
        )
    else:
        logger.warning("No candidate configuration passed validation")
    return TuningResult(
        best=best,
        trials=sorted(trials, key=lambda c: (not c.valid, *(-x for x in merit(c)))),
        rover=rover,
        base=base,
        config_text=config_text,
        validation=method,
        base_note=base_note,
        data_note=data_note,
    )
