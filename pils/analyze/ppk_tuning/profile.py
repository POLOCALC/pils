"""Profiles of the rover and base RINEX observations, and what limits fixing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

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
