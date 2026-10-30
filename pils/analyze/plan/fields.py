"""The fields a plan flies: what was flown where and when, where they are on the site and in the sky.

Every function takes the plan's tables (:class:`pils.analyze.plan.reader.PlanRun`) and
returns a plot-ready table.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import polars as pl


def _flown(schedule: pl.DataFrame) -> pl.DataFrame:
    """The schedule's telescope flights at a field."""
    return schedule.filter(
        pl.col("telescope").is_not_null() & pl.col("field").is_not_null() & (pl.col("field").cast(pl.Utf8) != "")
    )


def _counts(values: list[Any]) -> str:
    """"a 3, b 1": each value and how often, the most frequent first (ties: as they first appear)."""
    order: dict[Any, int] = {}
    for v in values:
        order[v] = order.get(v, 0) + 1
    ranked = sorted(order.items(), key=lambda kv: -kv[1])  # stable: ties keep their first appearance
    return ", ".join(f"{k} {n}" for k, n in ranked)


def field_summary(schedule: pl.DataFrame, fields: pl.DataFrame) -> pl.DataFrame:
    """One row per field flown, the most flights first: where it is (its reference telescope, az / el
    from it, its distance) and what was flown there -- flights, per telescope, per project, the days,
    the dates and the hours (first start to last end)."""
    rows = []
    for (field,), g in _flown(schedule).group_by("field", maintain_order=True):
        geo = fields.filter(pl.col("field") == field) if not fields.is_empty() else fields
        ref = geo["reference"][0] if "reference" in geo.columns and geo.height else None
        ref = ref if isinstance(ref, str) and ref else ""
        at_ref = geo.filter(pl.col("telescope") == ref) if ref else geo.clear()
        tels = g["telescope"].to_list()
        distance = (
            geo["distance_m"].cast(pl.Float64, strict=False).drop_nulls()
            if "distance_m" in geo.columns else pl.Series([], dtype=pl.Float64)
        )
        dates = sorted(g["date"].unique().to_list())
        rows.append({
            "field": field,
            "set from": ref or ("its own scan path" if set(tels) == {"LAT"} else "—"),
            "az / el": f"{at_ref['az_deg'][0]:.0f}° / {at_ref['el_deg'][0]:.0f}°" if at_ref.height else "—",
            "distance [m]": float(distance[0]) if len(distance) else None,
            "flights": g.height,
            "telescopes": _counts(tels),
            "projects": _counts(g["project"].to_list()),
            "days": len(dates),
            "dates": ", ".join(dt.date.fromisoformat(d).strftime("%d %b") for d in dates),
            "hours": f"{g['start'].min()}–{g['end'].max()}",
        })
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort("flights", descending=True, maintain_order=True)


def ground_points(schedule: pl.DataFrame, fields: pl.DataFrame, summary: pl.DataFrame) -> pl.DataFrame:
    """The fields of ``summary`` on the ground: E_m, N_m from the LAT (the spot under the drone), their
    reference telescope and their flights."""
    if summary.is_empty() or fields.is_empty():
        return pl.DataFrame()
    pos = fields.unique("field", keep="first", maintain_order=True)
    flights = schedule.filter(pl.col("telescope").is_not_null()).group_by("field").len()
    out = summary.select("field").join(pos, on="field", how="left").join(flights, on="field", how="left")
    reference = pl.col("reference").fill_null("") if "reference" in out.columns else pl.lit("")
    return out.select(
        "field", pl.col("E_m").cast(pl.Float64), pl.col("N_m").cast(pl.Float64),
        reference.alias("reference"), pl.col("len").fill_null(0).cast(pl.Int64).alias("flights"),
    )


def telescope_sites(config: dict[str, Any]) -> pl.DataFrame:
    """Every telescope of the configuration (its telescopes, then the observing ones not among them):
    telescope, E_m, N_m from the LAT."""
    from pils.geometry.site import site_enu

    sites: dict[str, tuple[float, float]] = {}
    for section in ("telescopes", "observing_telescopes"):
        for name, t in (config.get(section) or {}).items():
            site = (t or {}).get("site")
            if site is not None and name not in sites:
                e, n = (float(x) for x in site_enu(site)[:2])
                sites[name] = (e, n)
    return pl.DataFrame(
        {"telescope": list(sites), "E_m": [v[0] for v in sites.values()], "N_m": [v[1] for v in sites.values()]},
        schema={"telescope": pl.Utf8, "E_m": pl.Float64, "N_m": pl.Float64},
    )


def sky_positions(fields: pl.DataFrame, flown: list[str]) -> pl.DataFrame:
    """The fields flown as each telescope sees them (``fields.csv``): telescope, field, az, el, range,
    altitude, flights, envelope note -- for the polar map of one telescope's sky."""
    if fields.is_empty():
        return fields
    keep = [c for c in ("telescope", "field", "az_deg", "el_deg", "range_m", "altitude_m", "n_flights",
                        "envelope_note") if c in fields.columns]
    return fields.filter(pl.col("field").is_in(flown)).select(keep)


def field_days(schedule: pl.DataFrame) -> dict[str, Any]:
    """Field x day: the flights flown at each field each day -- ``fields`` (alphabetical), ``dates``
    (in order) and ``flights`` (one row per field, one count per date)."""
    flown = _flown(schedule)
    if flown.is_empty():
        return {"fields": [], "dates": [], "flights": []}
    fields = sorted(flown["field"].unique().to_list())
    dates = sorted(flown["date"].unique().to_list())
    counts = {(f, d): n for f, d, n in flown.group_by("field", "date").len().iter_rows()}
    return {"fields": fields, "dates": dates,
            "flights": [[int(counts.get((f, d), 0)) for d in dates] for f in fields]}
