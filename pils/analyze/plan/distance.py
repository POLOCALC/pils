"""How close an observing telescope's boresight comes to the drone, flight by flight.

A planner follows each flight flown for another telescope from an observing telescope
(the LAT), over every take-off time in the window and phase of its sweep: per flight
the worst, median and best closest approach, and the old field-centre test
(``distance_<TEL>.csv``, or computed at another scan speed). These functions shape
that table for its plots.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def followed(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The flights the telescope scanned during (a worst case), closest first."""
    got = [dict(r) for r in rows if r.get("worst_deg") is not None and not _nan(r.get("worst_deg"))]
    return sorted(got, key=lambda r: r["worst_deg"])


def _nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def summary(rows: Sequence[Mapping[str, Any]], exclusion: float) -> dict[str, Any]:
    """The flights followed, how many can come within the angle (worst case), do in the median case, the
    closest of all, and how many the field-centre test put within it."""
    got = followed(rows)
    return {
        "flights": len(got),
        "worst_within": sum(1 for r in got if r["worst_deg"] < exclusion),
        "median_within": sum(1 for r in got if r.get("median_deg") is not None and r["median_deg"] < exclusion),
        "closest_deg": min((r["worst_deg"] for r in got), default=None),
        "static_within": sum(1 for r in got if r.get("static_deg") is not None and r["static_deg"] < exclusion),
    }


def median_histogram(rows: Sequence[Mapping[str, Any]], bins: int = 30) -> dict[str, Any]:
    """The flights by their closest approach in the median case: ``bins`` bins from the closest flight to
    the farthest (none narrower than 0.05 deg), and the median over all flights."""
    v = sorted(r["median_deg"] for r in followed(rows) if r.get("median_deg") is not None)
    if not v:
        return {"edges": [], "counts": [], "median": None}
    lo, hi = v[0], v[-1]
    size = max((hi - lo) / bins, 0.05)
    n = int(math.floor((hi - lo) / size)) + 1
    counts = [0] * n
    for x in v:
        counts[min(n - 1, int(math.floor((x - lo) / size)))] += 1
    mid = len(v) // 2
    median = v[mid] if len(v) % 2 else (v[mid - 1] + v[mid]) / 2
    return {"edges": [lo + k * size for k in range(n + 1)], "counts": counts, "median": median}
