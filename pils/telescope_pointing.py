"""The telescopes' own pointing during a flight: the encoders of the SO SATs.

A flight's ``azel/<TELESCOPE>`` says where the drone was as each telescope saw it;
where the telescope itself looked is in the telescope's data -- for the SO SATs, the
ancillary stream of SO's books (``A_ancil_*.g3``: the azimuth, elevation and
boresight-rotation encoders at ~200 Hz, with their flags), kept in the campaigns'
``telescope_data/SO/data/<satpN>/obs/<ctime/1e5>/obs_<ctime>_<satpN>_<...>/``.
This reads the observations that overlap a flight and saves the pointing in the
flight's folder::

    proc/telescope/<TELESCOPE>/rev_YYYYMMDD_hhmmss/
        manifest.json, pointing.parquet     (kind "telescope_pointing")

with ``timestamp`` (s, UTC), ``az_enc_deg``, ``el_enc_deg``, ``boresight_enc_deg``
and ``flagged`` (any encoder flagged), at :data:`RATE_HZ`. The values are the
encoders' (an azimuth offset, the pointing model, are not applied). Reading the
books needs ``so3g`` (its ``spt3g``), imported only here.

    pils telescope <flight folder>... [-t SATP1] [--so-data <telescope_data/SO/data>]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils import store

logger = logging.getLogger(__name__)

# The SO SATs, as pils names them (azel/<TELESCOPE>) and as SO's books do
TELESCOPES = {"SATP1": "satp1", "SATP2": "satp2", "SATP3": "satp3"}
# The pointing kept, per second (the encoders run at ~200 Hz)
RATE_HZ = 20.0
OBS_PATTERN = re.compile(r"^obs_(\d+)_(satp\d)_")
DATASET = "telescope"
KIND = "telescope_pointing"


@dataclass
class PointingResult:
    """What was done for one telescope of one flight."""

    telescope: str
    status: str  # done, skipped, failed
    detail: str = ""
    path: Path | None = None
    rows: int = 0
    observations: list[str] = field(default_factory=list)


def so_data_root(flight_dir: Path) -> Path:
    """Return where SO's books are for a flight in <data>/campaigns/<campaign>/<day>/flight_*."""
    return Path(flight_dir).resolve().parents[2] / "telescope_data" / "SO" / "data"


def observations(root: Path, telescope: str, start: float, end: float) -> list[Path]:
    """Find the observations of a SAT (satpN) that may hold samples in [start, end].

    Those starting before ``end`` whose next one starts after ``start`` (UNIX s),
    by the start times in their names.
    """
    base = Path(root) / telescope / "obs"
    found: list[tuple[int, Path]] = []
    first = int(start // 1e5) - 1  # an observation may start in the previous folder
    for folder in sorted(base.glob("[0-9]*")) if base.is_dir() else []:
        if not folder.name.isdigit() or not (
            first <= int(folder.name) <= int(end // 1e5)
        ):
            continue
        for obs in folder.iterdir():
            match = OBS_PATTERN.match(obs.name)
            if match and obs.is_dir() and match.group(2) == telescope:
                found.append((int(match.group(1)), obs))
    found.sort()
    keep = []
    for i, (t0, obs) in enumerate(found):
        next_start = found[i + 1][0] if i + 1 < len(found) else float("inf")
        if t0 <= end and next_start > start:
            keep.append(obs)
    return keep


def read_ancil(files: list[Path], start: float, end: float) -> dict[str, np.ndarray]:
    """Read the encoders of ancillary files in [start, end].

    Returns the timestamp (s), az, el, boresight (deg) and whether flagged.
    """
    import so3g  # noqa: F401 - brings its spt3g (G3 files)
    from spt3g import core

    parts: dict[str, list[np.ndarray]] = {
        k: [] for k in ("timestamp", "az", "el", "boresight", "flagged")
    }
    for f in files:
        for frame in core.G3File(str(f)):
            if frame.type != core.G3FrameType.Scan or "ancil" not in frame:
                continue
            ancil = frame["ancil"]
            t = (
                np.array([x.time for x in ancil.times], dtype=np.float64)
                / core.G3Units.s
            )
            if not len(t) or t[-1] < start or t[0] > end:
                continue
            inside = (t >= start) & (t <= end)
            parts["timestamp"].append(t[inside])
            parts["az"].append(np.asarray(ancil["az_enc"])[inside])
            parts["el"].append(np.asarray(ancil["el_enc"])[inside])
            parts["boresight"].append(np.asarray(ancil["boresight_enc"])[inside])
            flags = [
                np.asarray(ancil[k], dtype=bool)[inside]
                for k in ("flag_az_enc", "flag_el_enc", "flag_boresight_enc")
                if k in ancil
            ]
            parts["flagged"].append(
                np.logical_or.reduce(flags) if flags else np.zeros(inside.sum(), bool)
            )
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in parts.items()}


def thin(samples: dict[str, np.ndarray], rate_hz: float = RATE_HZ) -> pl.DataFrame:
    """Put the samples in time order, at most ``rate_hz`` (a flagged sample stays flagged)."""
    order = np.argsort(samples["timestamp"], kind="stable")
    t = samples["timestamp"][order]
    keep = np.ones(len(t), bool)
    if len(t) > 1 and rate_hz > 0:
        slot = np.floor((t - t[0]) * rate_hz)
        keep = np.r_[True, np.diff(slot) > 0]
    pick = order[keep]
    return pl.DataFrame(
        {
            "timestamp": samples["timestamp"][pick],
            "az_enc_deg": samples["az"][pick].astype(np.float64),
            "el_enc_deg": samples["el"][pick].astype(np.float64),
            "boresight_enc_deg": samples["boresight"][pick].astype(np.float64),
            "flagged": samples["flagged"][pick].astype(bool),
        }
    )


def flight_window(flight_dir: Path) -> tuple[float, float] | None:
    """Return the flight's time span (UNIX s), from its synchronized drone table."""
    proc = Path(flight_dir) / "proc" / "flight" / "sync"
    if not proc.is_dir():
        return None
    rev = store.latest_version(proc)
    if not rev:
        return None
    directory = proc / rev
    tables = list(store.read_manifest(directory).get("tables") or {})
    table = "drone" if "drone" in tables else (tables[0] if tables else None)
    if table is None:
        return None
    t = store.read_columns(directory, table, ["timestamp"])["timestamp"].drop_nulls()
    return (float(t.min()), float(t.max())) if len(t) else None


def run(
    flight_dir: Path,
    telescopes: list[str] | None = None,
    so_data: Path | None = None,
    save: bool = True,
    margin_s: float = 30.0,
    rate_hz: float = RATE_HZ,
) -> list[PointingResult]:
    """Read the pointing of each SAT during a flight; save it in proc/telescope/<TELESCOPE>/."""
    flight_dir = Path(flight_dir)
    window = flight_window(flight_dir)
    names = [t.upper() for t in (telescopes or TELESCOPES)]
    if window is None:
        return [
            PointingResult(t, "failed", "no synchronized data (pils package first)")
            for t in names
        ]
    start, end = window[0] - margin_s, window[1] + margin_s
    root = Path(so_data) if so_data else so_data_root(flight_dir)
    version = store.new_version_name()
    out = []
    for name in names:
        so_name = TELESCOPES.get(name)
        if so_name is None:
            out.append(PointingResult(name, "skipped", "not an SO SAT"))
            continue
        obs = observations(root, so_name, start, end)
        if not obs:
            out.append(
                PointingResult(
                    name,
                    "skipped",
                    f"no observation of {so_name} in {root} during the flight",
                )
            )
            continue
        try:
            files = sorted(f for o in obs for f in o.glob("A_ancil_*.g3"))
            frame = thin(read_ancil(files, start, end), rate_hz)
        except Exception as e:  # a book that cannot be read: said, the others go on
            out.append(PointingResult(name, "failed", f"{type(e).__name__}: {e}"))
            continue
        if frame.is_empty():
            out.append(
                PointingResult(
                    name,
                    "skipped",
                    f"{so_name} recorded nothing during the flight",
                    observations=[o.name for o in obs],
                )
            )
            continue
        path = None
        if save:
            meta: dict[str, Any] = {
                "telescope_name": name,
                "so_telescope": so_name,
                "observations": [o.name for o in obs],
                "source": str(root),
                "flight_window": list(window),
                "margin_s": margin_s,
                "rate_hz": rate_hz,
                "values": "encoders (no azimuth offset, no pointing model)",
            }
            path = store.save_version(
                flight_dir / "proc" / DATASET,
                version,
                {"pointing": frame},
                meta,
                parent=name,
                kind=KIND,
            )
        out.append(
            PointingResult(
                name,
                "done",
                f"{len(frame)} samples from {len(obs)} observation(s)",
                path,
                len(frame),
                [o.name for o in obs],
            )
        )
    return out
