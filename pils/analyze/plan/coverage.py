"""The coverage a plan gives: which wafers and detectors its flights reach, how often, at which
rotations, and how long the drone stays in a beam.

Every function takes the plan's tables (:class:`pils.analyze.plan.reader.PlanRun`) and
returns plot-ready data: grids as their row and column labels and a list of rows, a
focal plane as one row per detector.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

ALL = "all projects"
# What a detector can be coloured by: {metric: its label}
METRICS = {
    "n_crossings": "crossings, primary + secondary",
    "n_crossings_primary": "crossings as primary",
    "n_crossings_secondary": "crossings as secondary",
    "n_flights": "flights illuminating it",
}


def _grid(df: pl.DataFrame, index: str, columns: str, rows: list, cols: list) -> list[list[int]]:
    """Rows of ``df`` counted per (index, columns), as a grid over ``rows`` x ``cols`` (0 where none)."""
    got = {(a, b): n for a, b, n in df.group_by(index, columns).len().iter_rows()} if df.height else {}
    return [[int(got.get((r, c), 0)) for c in cols] for r in rows]


def wafer_flights(schedule: pl.DataFrame, secondary: pl.DataFrame, project: str = ALL) -> dict[str, Any]:
    """Flights per telescope x wafer, one panel per project (or one, ``all projects``): the own flights
    (the schedule's) and the joins (``secondary.csv``'s joined rows), each a grid over ``telescopes`` x
    ``wafers``. ``has_joins``: the plan has a ``secondary.csv``."""
    cols = ["telescope", "wafer", "project"]
    own = schedule.select(cols).with_columns(pl.lit("own").alias("kind")) if set(cols) <= set(schedule.columns) else pl.DataFrame()
    parts = [own]
    has_joins = not secondary.is_empty() and "status" in secondary.columns
    if has_joins:
        parts.append(secondary.filter(pl.col("status") == "joined").select(cols).with_columns(pl.lit("joined").alias("kind")))
    flights = pl.concat([p.with_columns(pl.col(c).cast(pl.Utf8) for c in cols) for p in parts if p.height]) \
        if any(p.height for p in parts) else pl.DataFrame(schema={**{c: pl.Utf8 for c in cols}, "kind": pl.Utf8})
    projects_all = sorted(flights["project"].drop_nulls().unique().to_list())
    if project != ALL:
        flights = flights.filter(pl.col("project") == project)
    else:
        flights = flights.with_columns(pl.lit(ALL).alias("project"))
    df = flights.drop_nulls(cols).filter(pl.col("wafer") != "")
    projects = sorted(df["project"].unique().to_list())
    tels, wafers = sorted(df["telescope"].unique().to_list()), sorted(df["wafer"].unique().to_list())
    panels = []
    for p in projects:
        sub = df.filter(pl.col("project") == p)
        panels.append({
            "project": p,
            "own": _grid(sub.filter(pl.col("kind") == "own"), "telescope", "wafer", tels, wafers),
            "joined": _grid(sub.filter(pl.col("kind") == "joined"), "telescope", "wafer", tels, wafers),
        })
    return {"projects": projects_all, "telescopes": tels, "wafers": wafers, "panels": panels, "has_joins": has_joins}


def telescope_field_flights(schedule: pl.DataFrame) -> dict[str, Any]:
    """Flights per telescope x field: ``telescopes``, ``fields`` (alphabetical) and the counts."""
    df = schedule.drop_nulls(["telescope", "field", "date"]) if {"telescope", "field", "date"} <= set(schedule.columns) else pl.DataFrame()
    if df.is_empty():
        return {"telescopes": [], "fields": [], "flights": []}
    tels, fields = sorted(df["telescope"].unique().to_list()), sorted(df["field"].unique().to_list())
    return {"telescopes": tels, "fields": fields, "flights": _grid(df, "telescope", "field", tels, fields)}


def rotations_per_detector(
    rotations: pl.DataFrame, project: str = ALL, own_only: bool = False, source: str | None = None
) -> pl.DataFrame:
    """Every detector seen (``detector_rotations.csv``, its project, own flights or with the joins, its
    source): telescope, wafer, pixel, how many boresight rotations saw it and which ("+0 +30")."""
    sub = rotations
    if project != ALL:
        sub = sub.filter(pl.col("project") == project)
    if own_only:
        sub = sub.filter(pl.col("kind") == "own")
    if source is not None:
        sub = sub.filter(pl.col("source_ghz").cast(pl.Utf8) == str(source))
    if sub.is_empty():
        return pl.DataFrame(schema={"telescope": pl.Utf8, "wafer": pl.Utf8, "pixel": pl.Int64,
                                    "n_rotations": pl.Int64, "rotations": pl.Utf8})
    out = sub.group_by("telescope", "wafer", "pixel").agg(pl.col("boresight").cast(pl.Int64).unique().sort().alias("b"))
    return out.select(
        "telescope", "wafer", "pixel", pl.col("b").list.len().cast(pl.Int64).alias("n_rotations"),
        pl.col("b").map_elements(lambda s: " ".join(f"{int(x):+d}" for x in s), return_dtype=pl.Utf8).alias("rotations"),
    ).sort("telescope", "wafer", "pixel")


def rotations_summary(per_detector: pl.DataFrame, pixels: pl.DataFrame) -> dict[str, Any]:
    """Telescope x wafer: of all its detectors, how many were seen, at 2+ and at 3+ rotations, and the
    share at 2+ (the heatmap's colour): ``cells`` {telescope: {wafer: {...}}} over ``telescopes`` x ``wafers``."""
    if per_detector.is_empty():
        return {"telescopes": [], "wafers": [], "cells": {}}
    total = {(t, w): n for t, w, n in pixels.group_by("telescope", "wafer").len().iter_rows()}
    g = per_detector.group_by("telescope", "wafer").agg(
        pl.len().alias("seen"), (pl.col("n_rotations") >= 2).sum().alias("two"),
        (pl.col("n_rotations") >= 3).sum().alias("three"),
    )
    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for t, w, seen, two, three in g.iter_rows():
        n = total.get((t, w))
        cells.setdefault(t, {})[w] = {"seen": int(seen), "two": int(two), "three": int(three), "total": n,
                                      "share": (two / n) if n else None}
    return {"telescopes": sorted(cells), "wafers": sorted({w for c in cells.values() for w in c}), "cells": cells}


def rotations_map(per_detector: pl.DataFrame, pixels: pl.DataFrame, telescope: str) -> pl.DataFrame:
    """One telescope's detectors (``detector_pixels.csv``) with how many rotations saw each (0: never)."""
    pix = pixels.filter(pl.col("telescope") == telescope).select("wafer", "pixel", "xi_deg", "eta_deg")
    seen = per_detector.filter(pl.col("telescope") == telescope).select("wafer", "pixel", "n_rotations", "rotations")
    return pix.join(seen, on=["wafer", "pixel"], how="left").with_columns(
        pl.col("n_rotations").fill_null(0), pl.col("rotations").fill_null(""))


def section_flights(schedule: pl.DataFrame) -> pl.DataFrame:
    """Flights per telescope, wafer, section and boresight ("bs +30"): which slices were covered."""
    df = schedule.drop_nulls(["wafer", "section"])
    if df.is_empty():
        return pl.DataFrame(schema={"telescope": pl.Utf8, "wafer": pl.Utf8, "section": pl.Utf8,
                                    "boresight": pl.Utf8, "flights": pl.Int64})
    df = df.with_columns(
        pl.col("section").cast(pl.Float64).cast(pl.Int64).cast(pl.Utf8),
        pl.col("boresight").cast(pl.Float64).cast(pl.Int64)
        .map_elements(lambda b: f"bs {b:+d}", return_dtype=pl.Utf8).alias("boresight"),
    )
    return df.group_by("telescope", "wafer", "section", "boresight").len("flights") \
        .with_columns(pl.col("flights").cast(pl.Int64)).sort("telescope", "wafer", "section", "boresight")


def wafer_table(summary: pl.DataFrame) -> pl.DataFrame:
    """``wafer_summary.csv``'s wafers with own flights or joins."""
    if summary.is_empty():
        return summary
    return summary.filter((pl.col("own_flights") > 0) | (pl.col("joined_flights") > 0))


def _source_key(col: str = "source_ghz") -> pl.Expr:
    """90, 90.0 and "90": one source, as "90" ("" when none)."""
    v = pl.col(col).cast(pl.Float64, strict=False)
    return pl.when(v.is_null()).then(pl.lit("")).otherwise(
        v.map_elements(lambda x: f"{x:g}", return_dtype=pl.Utf8)).alias(col)


def with_secondaries(counts: pl.DataFrame, secondary: pl.DataFrame) -> pl.DataFrame:
    """The primary counts and the crossings collected as a secondary, side by side: n_crossings_primary,
    n_crossings_secondary and n_crossings, their total. A telescope that only ever joined gets rows of
    its own. Without secondary counts (or crossings), the counts as they are."""
    if secondary.is_empty() or "n_crossings" not in counts.columns:
        return counts
    key = ["telescope", "project", "source_ghz", "wafer", "pixel"]
    a = counts.with_columns(_source_key()).rename({"n_crossings": "n_crossings_primary"})
    b = secondary.with_columns(_source_key()).rename({"n_crossings": "n_crossings_secondary"}).select(
        *key, "n_crossings_secondary", pl.col("band").alias("band_sec"), pl.col("beam_arcmin").alias("beam_sec"))
    out = a.join(b, on=key, how="full", coalesce=True)
    out = out.with_columns(
        pl.col("band").fill_null(pl.col("band_sec")), pl.col("beam_arcmin").fill_null(pl.col("beam_sec")),
        *[pl.col(c).fill_null(0).cast(pl.Int64) for c in ("n_flights", "n_crossings_primary", "n_crossings_secondary")],
    ).drop("band_sec", "beam_sec")
    return out.with_columns((pl.col("n_crossings_primary") + pl.col("n_crossings_secondary")).alias("n_crossings"))


def detector_map(
    pixels: pl.DataFrame, counts: pl.DataFrame, telescope: str, project: str, source: str, metric: str = "n_flights"
) -> dict[str, Any]:
    """Every detector of a telescope with ``metric`` for one project and source (0: never reached), the
    other metrics beside it, the colour scale's top (the 98th percentile of the reached, rounded up),
    the reached and the beam."""
    pix = pixels.filter(pl.col("telescope") == telescope).select("wafer", "pixel", "xi_deg", "eta_deg")
    cnt = counts.filter((pl.col("telescope") == telescope) & (pl.col("project") == project)
                        & (pl.col("source_ghz").cast(pl.Utf8) == str(source)))
    metrics = [m for m in METRICS if m in cnt.columns]
    df = pix.join(cnt.select("wafer", "pixel", *metrics), on=["wafer", "pixel"], how="left").with_columns(
        pl.col(m).fill_null(0).cast(pl.Int64) for m in metrics)
    hit = df.filter(pl.col(metric) > 0)[metric].to_numpy() if metric in df.columns else np.zeros(0)
    beam = cnt["beam_arcmin"].drop_nulls() if "beam_arcmin" in cnt.columns else pl.Series([], dtype=pl.Float64)
    return {
        "detectors": df,
        "metrics": metrics,
        "vmax": int(np.ceil(np.percentile(hit, 98))) if len(hit) else 1,
        "reached": int(len(hit)),
        "beam_arcmin": float(beam[0]) if len(beam) else None,
    }


def wafer_histogram(
    counts: pl.DataFrame, telescope: str, project: str, source: str, wafer: str, metric: str = "n_flights"
) -> dict[str, Any]:
    """One wafer's detectors by how often they were reached over the campaign (``metric`` summed per
    detector): ``counts`` per value (0: never), and over the reached: how many, the median, the 10th-90th
    percentiles and the spread (std / mean)."""
    sub = counts.filter((pl.col("telescope") == telescope) & (pl.col("project") == project)
                        & (pl.col("source_ghz").cast(pl.Utf8) == str(source)) & (pl.col("wafer") == wafer))
    per = sub.group_by("pixel").agg(pl.col(metric).sum())[metric].to_numpy().astype(int) if sub.height else np.zeros(0, int)
    hit = per[per > 0]
    out: dict[str, Any] = {"counts": np.bincount(per).tolist() if per.size else [0], "detectors": int(per.size),
                           "reached": int(hit.size)}
    if hit.size:
        p10, p90 = np.percentile(hit, [10, 90])
        out.update(median=float(np.median(hit)), p10=float(p10), p90=float(p90),
                   spread=float(hit.std() / hit.mean()) if hit.mean() else 0.0)
    return out


def crossing_times(
    times: pl.DataFrame, telescope: str, project: str, source: str, hwp_hz: float | None,
    beam_fraction: float = 0.5,
) -> dict[str, Any]:
    """How long the drone stays in a beam, for one telescope, project (or all) and source: two histograms
    on one time axis -- every crossing (a row stands for its ``n_flights``) and every detector by its
    longest crossing -- with the HWP demodulation's limit, and the numbers: crossings, median time,
    detectors crossed, the share crossed at least once for the limit."""
    from pils.geometry.crossings import demod_limit_s

    sub = times.filter(pl.col("telescope") == telescope)
    if project != ALL:
        sub = sub.filter(pl.col("project") == project)
    sub = sub.filter(pl.col("source_ghz").cast(pl.Utf8) == str(source))
    if sub.is_empty():
        return {"empty": True}
    limit = demod_limit_s(hwp_hz, beam_fraction)
    d = sub["crossing_s"].to_numpy().astype(float)
    w = sub["n_flights"].to_numpy().astype(int)
    longest = sub.group_by("wafer", "pixel").agg(pl.col("crossing_s").max())["crossing_s"].to_numpy().astype(float)
    every = np.repeat(d, w)
    hi = max(float(np.percentile(every, 99.5)) if len(d) else 1.0, limit or 0.0) * 1.05
    steps = np.diff(np.unique(d))  # the times come in steps of the path's sampling: no bin narrower than one
    width = max(hi / 40, float(steps[steps > 1e-9].min()) if (steps > 1e-9).any() else 0.0)
    edges = np.arange(0, hi + width, width)
    n_cross, _ = np.histogram(np.clip(d, 0, hi), bins=edges, weights=w)
    n_det, _ = np.histogram(np.clip(longest, 0, hi), bins=edges)
    return {
        "empty": False,
        "edges": edges.tolist(),
        "crossings": n_cross.astype(int).tolist(),
        "detectors": n_det.astype(int).tolist(),
        "limit_s": limit,
        "hwp_hz": hwp_hz,
        "n_crossings": int(len(every)),
        "median_s": float(np.median(every)) if len(every) else None,
        "n_detectors": int(len(longest)),
        "share_long_enough": float((longest >= limit).mean()) if limit else None,
        "beam_arcmin": float(sub["beam_arcmin"][0]) if "beam_arcmin" in sub.columns else None,
    }
