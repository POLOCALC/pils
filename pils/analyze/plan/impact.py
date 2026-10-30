"""The impact a plan has on the other telescopes: which of their scans the drone disturbs, for how
long, and what would change it.

A planner writes, per telescope, ``impact_<TEL>.csv`` (every scan's affected minutes under
scenarios of azimuth trim and exclusion angle; ``TOTAL`` and ``ALL <priority>`` rows) and
``impact_days_<TEL>.csv`` (the scans disturbed, day by day). Every function returns
plot-ready data, at the exclusion angle the plan used (:func:`planned_exclusion`) unless it
says otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

PRIORITY_ORDER = ("High", "Low", "Calibration", "Source", "Other")


def planned_exclusion(rows: Sequence[Mapping]) -> float:
    """The exclusion angle a table of impact rows was actually planned against.

    The scenario sweep carries the *other* rule's angle too, so the in-force one
    is not simply the largest -- under ``telescope_beam`` it is the smaller of the
    two. It is the one the trim scenarios were run at, because only the planned
    angle is swept in trim.
    """
    planned = {float(r["exclusion_deg"]) for r in rows if float(r["trim_deg"]) > 0}
    return planned.pop() if len(planned) == 1 else max(float(r["exclusion_deg"]) for r in rows)


def exclusion_of(impact: pl.DataFrame) -> float:
    """:func:`planned_exclusion` of an impact table."""
    return planned_exclusion(impact.select("trim_deg", "exclusion_deg").to_dicts())


def _planned(impact: pl.DataFrame, exclusion: float) -> pl.DataFrame:
    return impact.filter((pl.col("trim_deg") == 0) & (pl.col("exclusion_deg") == exclusion))


def _is_summary(col: str = "target") -> pl.Expr:
    return (pl.col(col) == "TOTAL") | pl.col(col).cast(pl.Utf8).str.starts_with("ALL ")


def total(impact: pl.DataFrame, exclusion: float) -> dict[str, Any] | None:
    """The telescope's whole observing at the planned angle: the ``TOTAL`` row (affected, observed, the
    share, the flights that affect it)."""
    rows = _planned(impact, exclusion).filter(pl.col("target") == "TOTAL")
    return rows.row(0, named=True) if rows.height else None


def by_priority(impact: pl.DataFrame, exclusion: float) -> pl.DataFrame:
    """Per priority class (High, Low, Calibration, Source, Other, those it has): affected and unaffected
    minutes (observed less affected, never below 0: the charge is wall clock) and the share."""
    df = _planned(impact, exclusion).filter(pl.col("target").cast(pl.Utf8).str.starts_with("ALL "))
    df = df.with_columns(pl.col("target").str.replace("ALL ", "", literal=True).alias("class"))
    order = [c for c in PRIORITY_ORDER if c in set(df["class"].to_list())]
    df = df.filter(pl.col("class").is_in(order)).with_columns(
        pl.col("class").replace_strict({c: i for i, c in enumerate(order)}, return_dtype=pl.Int64).alias("_o"))
    return df.sort("_o").select(
        "class", "affected_min", "observed_min",
        (pl.col("observed_min") - pl.col("affected_min")).clip(lower_bound=0).alias("unaffected_min"),
        "affected_frac_of_target", "n_flights_affected",
    )


def by_scan(impact: pl.DataFrame, exclusion: float) -> pl.DataFrame:
    """Per scan (no ``TOTAL``, no ``ALL``), as the impact table lists them: affected, unaffected (never
    below 0) and observed minutes, the share, the flights."""
    df = _planned(impact, exclusion).filter(~_is_summary())
    return df.select(
        "target", *[c for c in ("priority",) if c in df.columns], "affected_min", "observed_min",
        (pl.col("observed_min") - pl.col("affected_min")).clip(lower_bound=0).alias("unaffected_min"),
        "affected_frac_of_target", "n_flights_affected",
    )


def scenarios(impact: pl.DataFrame, exclusion: float) -> dict[str, pl.DataFrame]:
    """What would change it, from the ``TOTAL`` rows: ``trim`` -- trimming each end of every scan, at
    the planned angle -- and ``exclusion`` -- other angles, untrimmed: x, the share, the affected
    minutes and the flights."""
    t = impact.filter(pl.col("target") == "TOTAL")
    cols = ["affected_frac_of_target", "affected_min", "n_flights_affected"]
    return {
        "trim": t.filter(pl.col("exclusion_deg") == exclusion).sort("trim_deg").select("trim_deg", *cols),
        "exclusion": t.filter(pl.col("trim_deg") == 0.0).sort("exclusion_deg").select("exclusion_deg", *cols),
    }


def closest_by_day(days: pl.DataFrame) -> dict[str, dict[str, Any]]:
    """{scan: its day-by-day row where it came closest to the drone} (``impact_days_<TEL>.csv``)."""
    if days.is_empty() or "min_separation_deg" not in days.columns:
        return {}
    best = days.with_row_index("_i").sort("min_separation_deg", "_i").unique("target", keep="first", maintain_order=True)
    return {r["target"]: r for r in best.drop("_i").to_dicts()}


def when(approach: Mapping[str, Any] | None, exclusion: float) -> str:
    """Where and when a scan came closest to the drone, and whether that was inside the angle."""
    if not approach:
        return "not observing while the drone flew"
    inside = "inside" if approach["closest_deg"] < exclusion else "outside"
    return (f"{approach['date']} {approach['time']}, {approach['field']} (for {approach['flown_for']}) "
            f"— {inside} the {exclusion:.1f}° angle")


def scan_rows(
    impact: pl.DataFrame, exclusion: float, days: pl.DataFrame, approach: Mapping[str, Mapping] | None,
    recorded: bool,
) -> list[dict[str, Any]]:
    """Every scan at the planned angle with how close it came to the drone and when: from the closest
    approach over the whole campaign (``approach``, {scan: row}, inside the angle or not) when known,
    else from the day-by-day table (only the scans inside the angle). ``recorded``: the run wrote
    day-by-day tables at all (older runs did not)."""
    near = closest_by_day(days)
    approach = approach or {}
    out = []
    for x in _planned(impact, exclusion).filter(~_is_summary()).to_dicts():
        a, c = approach.get(x["target"]), near.get(x["target"])
        out.append({
            "scan": x["target"], "priority": x.get("priority") or "—",
            "affected_min": x["affected_min"], "observed_min": x["observed_min"],
            "share": x["affected_frac_of_target"], "flights": x["n_flights_affected"],
            "closest_deg": a["closest_deg"] if a else (c or {}).get("min_separation_deg"),
            "when": (when(a, exclusion) if a else f"{c['date']}, {c['fields']}" if c
                     else f"never within {exclusion:.1f}°" if recorded else "not recorded — older run"),
        })
    return out


def rule_text(config: Mapping[str, Any], name: str) -> str:
    """How the run's exclusion angle for ``name`` was made, in words, from the configuration it used."""
    spec = ((config.get("observing_telescopes") or {}).get(name) or (config.get("telescopes") or {}).get(name) or {})
    if spec.get("exclusion_deg") not in (None, ""):
        return f"its own `exclusion_deg`, {float(spec['exclusion_deg']):g}° set for {name}"
    ds = config.get("drone_source") or {}
    rule = ds.get("exclusion_rule", "source_beam")
    if rule == "fixed":
        return f"the `fixed` rule, {ds.get('fixed_deg')}° set by hand for every telescope"
    if rule == "telescope_beam":
        return f"the `telescope_beam` rule, {name}'s field of view / 2 + {ds.get('n_beams', 5)} of its beams"
    return (f"the `source_beam` rule, {ds.get('safety_factor', 1.5)} × the {ds.get('beam_fwhm_deg')}° source beam / 2 "
            f"+ {name}'s field of view / 2")
