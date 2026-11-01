"""Candidate RTKLIB configurations (a grid from the data profile, random ones) and template edits."""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from pils.analyze.ppk_tuning.profile import (
    MIN_COMMON_SATS,
    MIN_L2_SNR,
    NAVSYS_BITS,
    ObsProfile,
)

logger = logging.getLogger(__name__)


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
