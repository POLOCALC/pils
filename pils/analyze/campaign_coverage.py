"""A campaign's calibration so far: per telescope, band and detector, the crossings of all its flights.

Each flight's coverage of each telescope it has the pointing of (:func:`pils.analyze.coverage.
telescope_coverage`: ``proc/azel/<T>`` and ``proc/telescope/<T>``), at the flight's source, added
up per detector -- the detectors in the focal plane's own frame (no boresight rotation), so the
same detector across flights with different rotations. With it the boresight rotations flown, and
per wafer the detectors crossed at least once and at least :data:`ENOUGH` times.

What this does not have yet: a polarisation angle per detector, which needs the detectors' data.

    campaign_coverage(flights, telescopes) -> dict
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

# Crossings of a detector counted as enough (several flights, several rotations)
ENOUGH = 3


def _layout(description: Mapping[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """List the telescope's pixels unrotated: wafer, xi, eta (degrees), in coverage order."""
    from pils.geometry.detectors import rotated_pixels
    from pils.geometry.focal_plane import wafer_from_dict

    wafers = [wafer_from_dict(w) for w in description["wafers"]]
    names: list[str] = []
    xs: list[float] = []
    ys: list[float] = []
    for wafer, xy in rotated_pixels(wafers, description["wafer_type"], 0.0).items():
        names += [wafer] * len(xy)
        xs += np.round(xy[:, 0], 4).tolist()
        ys += np.round(xy[:, 1], 4).tolist()
    return names, np.array(xs), np.array(ys)


def flight_telescopes(flight_dir: str | Path) -> list[str]:
    """List the telescopes a flight has the pointing of and saw the drone from."""
    proc = Path(flight_dir) / "proc"
    pointing = proc / "telescope"
    if not pointing.is_dir():
        return []
    return sorted(
        p.name
        for p in pointing.iterdir()
        if p.is_dir() and (proc / "azel" / p.name).is_dir()
    )


def campaign_coverage(
    flights: Iterable[Mapping[str, Any]],
    telescopes: Mapping[str, Mapping[str, Any]],
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Add up a campaign's flights' coverage, per telescope and band.

    Parameters
    ----------
    flights : iterable of mapping
        ``name``, ``path`` (the flight's folder) and ``source_ghz`` (None: skipped, said).
    telescopes : mapping
        Per telescope name (upper case), its description
        (:func:`pils.analyze.coverage.describe_telescope`); others are skipped.
    progress : callable, optional
        ``progress(done, total)`` after each flight and telescope.

    Returns
    -------
    dict
        ``targets``: per ``<TELESCOPE> <band>``, the pixels (``wafer``, ``xi``, ``eta``
        unrotated, ``crossings`` summed, ``flights`` crossing), per wafer the detectors,
        those crossed and crossed :data:`ENOUGH` times, the rotations flown and the flights;
        ``flights``: per flight and telescope, what was found or why not; ``enough``.
    """
    from pils.analyze.coverage import telescope_coverage

    flights = list(flights)
    work = [
        (f, t) for f in flights for t in flight_telescopes(f["path"]) if t in telescopes
    ]
    layouts: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    targets: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for k, (flight, telescope) in enumerate(work):
        row: dict[str, Any] = {"flight": flight["name"], "telescope": telescope}
        source = flight.get("source_ghz")
        try:
            if source is None:
                raise ValueError("no source frequency recorded")
            cov = telescope_coverage(
                flight["path"], dict(telescopes[telescope]), float(source)
            )
        except (ValueError, KeyError, OSError) as e:
            detail = str(e).splitlines()[0][:200]
            # A telescope without the source's band: nothing to see, not a problem
            status = "no band" if detail.startswith("no band of") else "skipped"
            rows.append({**row, "status": status, "detail": detail})
        else:
            if telescope not in layouts:
                layouts[telescope] = _layout(telescopes[telescope])
            wafers, xi, eta = layouts[telescope]
            counts = np.asarray(cov["pixels"]["crossings"], dtype=int)
            key = f"{telescope} {cov['band']}"
            target = targets.setdefault(
                key,
                {
                    "telescope": telescope,
                    "band": cov["band"],
                    "beam_fwhm_arcmin": cov["beam_fwhm_arcmin"],
                    "crossings": np.zeros(len(wafers), dtype=int),
                    "flights_crossing": np.zeros(len(wafers), dtype=int),
                    "rotations": {},
                    "flights": [],
                },
            )
            target["crossings"] += counts
            target["flights_crossing"] += counts > 0
            rotation = f"{round(cov['boresight_deg'], 1) + 0.0:g}"  # 0.1 deg; no "-0"
            target["rotations"][rotation] = target["rotations"].get(rotation, 0) + 1
            target["flights"].append(flight["name"])
            rows.append(
                {
                    **row,
                    "status": "done",
                    "band": cov["band"],
                    "crossed": cov["crossed"],
                    "boresight_deg": cov["boresight_deg"],
                }
            )
        if progress is not None:
            progress(k + 1, len(work))

    out_targets = {}
    for key, target in sorted(targets.items()):
        wafers, xi, eta = layouts[target["telescope"]]
        counts, flights_crossing = target["crossings"], target["flights_crossing"]
        per_wafer = {}
        for wafer in dict.fromkeys(wafers):
            mask = np.array([w == wafer for w in wafers])
            per_wafer[wafer] = {
                "detectors": int(mask.sum()),
                "crossed": int((counts[mask] > 0).sum()),
                "enough": int((counts[mask] >= ENOUGH).sum()),
                "crossings": int(counts[mask].sum()),
            }
        out_targets[key] = {
            "telescope": target["telescope"],
            "band": target["band"],
            "beam_fwhm_arcmin": target["beam_fwhm_arcmin"],
            "flights": target["flights"],
            "rotations": dict(
                sorted(target["rotations"].items(), key=lambda r: float(r[0]))
            ),
            "wafers": per_wafer,
            "pixels": {
                "wafer": wafers,
                "xi": xi.tolist(),
                "eta": eta.tolist(),
                "crossings": counts.tolist(),
                "flights": flights_crossing.tolist(),
            },
        }
    return {"targets": out_targets, "flights": rows, "enough": ENOUGH}
