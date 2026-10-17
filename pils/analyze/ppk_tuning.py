"""Tune the RTKLIB configuration to the data of a flight.

The best RTKLIB settings depend on the data: which signals the rover tracks
well (a payload antenna may have a weak L2), which constellations both
receivers see, how fast the data are. :func:`tune` profiles the rover and base
RINEX files, builds candidate configurations from the profile around a
template, runs them all with ``rnx2rtkp`` in parallel, and keeps the one with
the most fixed epochs among those that pass validation:

- with a reference track (the drone's RTK solution, flown about a metre from
  the payload antenna), the fixed epochs must follow it: after removing the
  median offset (lever arm), horizontal and vertical RMS below
  ``max_h_rms``/``max_v_rms`` and at most ``max_bad_pct`` % of the fixed
  epochs off by more than 1 m;
- without one, the forward and backward solutions must agree on the epochs
  that both fix; this check is weaker (both directions can hold the same
  wrong ambiguities with fix-and-hold, which is then not tried).

The fix rate alone is not a safe criterion: lower ratio thresholds and
fix-and-hold raise it with wrong fixes (on flight_20251206_1833, 72 % fixed
but 1.5 % of the fixes off by more than a metre, against 65 % with none).

Examples
--------
>>> result = tune(template, rover_obs, base_obs, nav, workdir,
...               reference=drone_track)  # doctest: +SKIP
>>> result.best.overrides, result.best.fix_pct  # doctest: +SKIP
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import shutil
import subprocess
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
    valid: bool = False
    note: str = ""

    def row(self) -> dict[str, Any]:
        return {
            "name": self.name,
            **{f"set_{k}": v for k, v in self.overrides.items()},
            "epochs": self.epochs,
            "fix_pct": self.fix_pct,
            "h_rms_m": self.h_rms,
            "v_rms_m": self.v_rms,
            "bad_pct": self.bad_pct,
            "valid": self.valid,
            "note": self.note,
        }


def snr_masks(rover: ObsProfile, template: str = "") -> list[tuple[str, int | None]]:
    """SNR masks to try: the template's (if on), one from the rover's L1 SNR
    (median of GPS and Galileo minus 5 dB-Hz, within 20-35), and none.

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
MAX_BASE_OFFSET_M = 50.0


def config_value(text: str, key: str) -> str | None:
    """A setting of an RTKLIB config text (without its comment)."""
    match = re.search(rf"^{re.escape(key)}\s*=([^#\n]*)", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def base_position_check(text: str, base: ObsProfile) -> tuple[dict[str, str], str]:
    """Overrides if the template's base position does not match the base data.

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
    """The template config text with some settings replaced (or added)."""
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
    """A .pos file as a table (schema names, UTC timestamp)."""
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

    def table(self) -> pl.DataFrame:
        return pl.DataFrame([c.row() for c in self.trials])

    def metadata(self) -> dict[str, Any]:
        return {
            "tuning_validation": self.validation,
            "tuning_base_position": self.base_note,
            "tuning_data_quality": self.data_note,
            "tuning_best": self.best.row() if self.best else None,
            "tuning_trials": len(self.trials),
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
        ``lat_deg``, ``lon_deg``, ``h_ell_m``), e.g. the drone RTK track.
        Without it, forward and backward solutions are compared (two runs
        per candidate).
    max_h_rms, max_v_rms : float
        Largest RMS (m) of the fixed epochs against the reference.
    max_bad_pct : float
        Largest % of fixed epochs off by more than 1 m (0.2 m between
        directions without a reference).
    workers : int, optional
        Parallel runs (default: CPU count).

    Returns
    -------
    TuningResult
        The best candidate (most fixed epochs among the valid ones; None if
        none is valid), every trial, the data profiles and the chosen config.
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
    directions = reference is None
    # Forward and backward fix-and-hold solutions can hold the same wrong
    # ambiguities: without a reference, only continuous resolution is tried
    trials = candidates(rover, base, hold=not directions, template=text)
    method = "forward/backward" if directions else "reference track"
    logger.info(
        f"Tuning RTKLIB: {len(trials)} candidates, validation "
        + ("forward/backward" if directions else "reference track")
    )

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
                assert reference is not None
                check = (
                    compare_to_reference(solution, reference)
                    if solution.height
                    else None
                )
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
    best = (
        max(valid, key=lambda c: (round(c.fix_pct, 1), -(c.h_rms or 0)))
        if valid
        else None
    )
    config_text = None
    if best is not None:
        config_text = apply_overrides(text, best.overrides)
        logger.info(
            f"Best RTKLIB config {best.name}: {best.fix_pct:.1f} % fixed, "
            f"h RMS {best.h_rms:.3f} m, v RMS {best.v_rms:.3f} m"
        )
    else:
        logger.warning("No candidate configuration passed validation")
    return TuningResult(
        best=best,
        trials=sorted(trials, key=lambda c: (-c.valid, -c.fix_pct)),
        rover=rover,
        base=base,
        config_text=config_text,
        validation=method,
        base_note=base_note,
        data_note=data_note,
    )
