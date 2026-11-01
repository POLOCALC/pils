"""RINEX observations for the quick look: epochs, SNR per band, common satellites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pils.analyze.quicklook.common import (
    SNR_LEVELS,
)
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


# -- RINEX -------------------------------------------------------------------


def read_rinex(path: str | Path) -> tuple[list[str], list[dict[str, dict]]]:
    """Epoch times and, per epoch, ``{"G05": {"1": (snr, lli), ...}}``.

    SNR is None when missing; LLI is the loss-of-lock bit of the band's phase.
    """
    types: dict[str, list[str]] = {}
    system = ""
    with open(path, errors="replace") as fh:
        for line in fh:
            label = line[60:].strip()
            if label == "SYS / # / OBS TYPES":
                if line[0] != " ":
                    system = line[0]
                    types[system] = []
                types[system] += line[7:58].split()
            if label == "END OF HEADER":
                break
        times: list[str] = []
        epochs: list[dict[str, dict]] = []
        current: dict[str, dict] = {}
        for line in fh:
            if line.startswith(">"):
                current = {}
                epochs.append(current)
                times.append(line[2:21])
                continue
            obs = types.get(line[0])
            if obs is None or not epochs:
                continue
            bands: dict[str, list] = {}
            for j, kind in enumerate(obs):
                if kind[0] not in "LS":
                    continue
                text = line[3 + 16 * j : 3 + 16 * j + 16]
                if not text[:14].strip():
                    continue
                entry = bands.setdefault(kind[1], [None, 0])
                if kind[0] == "S":
                    value = float(text[:14])
                    entry[0] = value if value > 0 else None
                else:
                    flag = text[14:15].strip()
                    entry[1] = int(flag) & 1 if flag.isdigit() else 0
            current[line[:3]] = {b: tuple(v) for b, v in bands.items()}
    return times, epochs


def band_table(
    epochs: list[dict[str, dict]], interval_s: float = 1.0
) -> list[dict[str, Any]]:
    """Summarize the RINEX observations per system and band.

    Satellites (all, and above each SNR level), SNR, loss of lock and
    continuous tracking (observation-weighted arc, s).
    """
    keys = sorted(
        {
            (sat[0], band)
            for epoch in epochs
            for sat, obs in epoch.items()
            for band in obs
        }
    )
    rows = []
    for system, band in keys:
        counts: dict[int, list[int]] = {level: [] for level in (0, *SNR_LEVELS)}
        snrs: list[float] = []
        flagged = observed = 0
        running: dict[str, int] = {}
        arcs: list[int] = []
        for epoch in epochs:
            values = []
            seen = set()
            for sat, obs in epoch.items():
                if sat[0] != system or band not in obs or obs[band][0] is None:
                    continue
                snr, lli = obs[band]
                values.append(snr)
                observed += 1
                flagged += lli
                seen.add(sat)
            for level in counts:
                counts[level].append(sum(v >= level for v in values))
            snrs += values
            for sat in list(running):
                if sat not in seen:
                    arcs.append(running.pop(sat))
            for sat in seen:
                running[sat] = running.get(sat, 0) + 1
        arcs += running.values()
        if not snrs:
            continue
        row: dict[str, Any] = {"system": system, "band": band}
        row["sats"] = float(np.median(counts[0]))
        for level in SNR_LEVELS:
            row[f"sats_{level}dbhz"] = float(np.median(counts[level]))
        row["snr_dbhz"] = float(np.median(snrs))
        row["snr_p10_dbhz"] = float(np.percentile(snrs, 10))
        row["lli_pct"] = 100 * flagged / observed if observed else float("nan")
        row["arc_s"] = (
            float(np.sum(np.square(arcs)) / np.sum(arcs)) * interval_s if arcs else 0.0
        )
        rows.append(row)
    return rows


def common_satellites(
    rover: tuple[list[str], list[dict[str, dict]]],
    base: tuple[list[str], list[dict[str, dict]]],
) -> dict[str, float]:
    """Median satellites seen by both receivers at the same epoch, by system."""
    base_by_time = {t[:19]: epoch for t, epoch in zip(*base, strict=True)}
    counts: dict[str, list[int]] = {}
    for t, epoch in zip(*rover, strict=True):
        other = base_by_time.get(t[:19])
        if other is None:
            continue
        both = set(epoch) & set(other)
        for system in {s[0] for s in epoch} | {s[0] for s in other}:
            counts.setdefault(system, []).append(sum(s[0] == system for s in both))
    return {s: float(np.median(v)) for s, v in sorted(counts.items())}


def _rinex_time(text: str) -> np.datetime64:
    parts = text.split()
    year, month, day, hour, minute = (int(p) for p in parts[:5])
    second = int(float(parts[5]))
    return np.datetime64(
        f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
    )
