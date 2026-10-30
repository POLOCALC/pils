"""The campaign day by day: the Gantt's flights and empty blocks, a project's shared window, the flights.

Every function takes the plan's tables (:class:`pils.analyze.plan.reader.PlanRun`) and
returns a plot-ready table: one row per mark, times as the schedule's local ``HH:MM``.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from pils.analyze.plan.reader import project_name

# What a Gantt can be coloured by: {key: its label}
COLOUR_BY = {
    "project": "project, per telescope",
    "project_total": "project, total",
    "telescope": "telescope",
    "field": "field",
}
# The reasons a block is empty (the planner's gaps.csv), in the order they are drawn and listed
GAP_REASONS = [
    "Sun", "protected scan", "field change", "nothing to fly", "prep", "no-fly", "blackout", "free",
]
# What a flight's box tells when hovered, of the schedule's columns
GANTT_DETAILS = (
    "telescope", "field", "wafer", "section", "boresight", "source", "sat_az_min", "sat_az_max",
    "sat_el", "drone_el_lo", "drone_el_hi", "schedule_conflict",
)
# The flights table's columns, in order
FLIGHT_COLUMNS = (
    "flight", "date", "start", "end", "telescope", "project", "field", "wafer", "boresight", "section",
    "source", "sat_az_min", "sat_az_max", "sat_el", "drone_heading_deg", "telescope_speed_deg_s",
    "drone_speed_deg_s", "band_crossed", "drone_el_lo", "drone_el_hi", "drone_alt_hi_m",
    "drone_range_m", "above_ceiling", "illuminated", "schedule_conflict",
)


def _text(df: pl.DataFrame, column: str) -> pl.Expr:
    return (pl.col(column).cast(pl.Utf8) if column in df.columns else pl.lit(None, pl.Utf8)).fill_null("—")


def gantt_flights(schedule: pl.DataFrame, colour_by: str = "project") -> pl.DataFrame:
    """One box per block of the schedule: date, start, end, its flight, its label and colour key.

    The label is ``wafer.section`` (``3.1``: ws3, its band 1); the key is what the boxes are
    coloured by (:data:`COLOUR_BY`): the project per telescope (``POLOCALC/satp1``), the project
    alone, the telescope or the field. The hover's columns (:data:`GANTT_DETAILS`) follow.
    """
    if colour_by not in COLOUR_BY:
        raise ValueError(f"colour by one of {', '.join(COLOUR_BY)}, not {colour_by!r}")
    if schedule.is_empty():
        return pl.DataFrame()
    wafer = (
        pl.col("wafer").cast(pl.Utf8).fill_null("").str.strip_prefix("ws")
        if "wafer" in schedule.columns else pl.lit("")
    )
    section = (
        pl.col("section").cast(pl.Float64, strict=False)
        if "section" in schedule.columns else pl.lit(None, pl.Float64)
    )
    label = wafer + pl.when(section.is_null()).then(pl.lit("")).otherwise(
        pl.lit(".") + section.cast(pl.Int64).cast(pl.Utf8)
    )
    key = _text(schedule, "project" if colour_by == "project_total" else colour_by)
    if colour_by == "project" and "telescope" in schedule.columns:
        key = key + pl.lit("/") + _text(schedule, "telescope")
    details = [c for c in GANTT_DETAILS if c in schedule.columns]
    flight = pl.col("flight") if "flight" in schedule.columns else pl.lit(None, pl.Int64)
    return schedule.select(
        "date", "start", "end", flight.alias("flight"), label.alias("label"), key.alias("key"),
        *[c for c in ("project",) if c in schedule.columns and c not in details], *details,
    )


def colour_keys(flights: pl.DataFrame) -> list[str]:
    """The colour keys of :func:`gantt_flights`, in the order they first appear (the legend's)."""
    return [] if flights.is_empty() else flights["key"].unique(maintain_order=True).to_list()


def days(schedule: pl.DataFrame) -> list[str]:
    """The days of the schedule, in order: the Gantt's lines."""
    return [] if schedule.is_empty() else sorted(schedule["date"].drop_nulls().unique().to_list())


def gap_bands(gaps: pl.DataFrame) -> pl.DataFrame:
    """The empty blocks (``gaps.csv``): date, start, end, reason, minutes -- in the order of
    :data:`GAP_REASONS` (the ones drawn first lie behind)."""
    if gaps.is_empty():
        return pl.DataFrame()
    order = pl.DataFrame({"reason": GAP_REASONS, "_order": range(len(GAP_REASONS))})
    return (
        gaps.with_row_index("_row")
        .join(order, on="reason", how="inner")
        .sort("_order", "_row")
        .select("date", "start", "end", "reason", "minutes")
    )


def gap_summary(gaps: pl.DataFrame) -> pl.DataFrame:
    """The empty blocks by reason: minutes in all and holes, the most minutes first."""
    if gaps.is_empty():
        return pl.DataFrame()
    return (
        gaps.group_by("reason", maintain_order=True)
        .agg(pl.col("minutes").sum().alias("minutes"), pl.len().alias("holes"))
        .sort("minutes", descending=True, maintain_order=True)
    )


def _clock(hhmm: str, plus_minutes: int = 0) -> str:
    h, m = (int(x) for x in str(hhmm).split(":")[:2])
    total = h * 60 + m + plus_minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def sun_blocked(
    schedule: pl.DataFrame, availability: pl.DataFrame, block_minutes: int = 30
) -> pl.DataFrame:
    """Per day, the stretches when the Sun leaves no wafer usable for the telescope and field
    flown that day (``availability.csv``'s ``n_safe_sections`` 0): date, start, end.

    The background of a Gantt when the plan has no ``gaps.csv`` -- "safe for some target
    somewhere" would be true almost all day.
    """
    if schedule.is_empty() or availability.is_empty():
        return pl.DataFrame()
    flown = (
        schedule.drop_nulls(["telescope", "field"])
        .group_by("date", maintain_order=True)
        .agg(pl.col("telescope").first(), pl.col("field").first())
        .sort("date")
    )
    out: list[dict[str, Any]] = []
    for date, telescope, field in flown.iter_rows():
        day = availability.filter(
            (pl.col("date") == date) & (pl.col("telescope") == telescope) & (pl.col("field") == field)
        ).sort("block")
        start = None
        for time, safe in day.select("time", "n_safe_sections").iter_rows():
            if safe == 0 and start is None:
                start = time
            elif safe != 0 and start is not None:
                out.append({"date": date, "start": start, "end": time})
                start = None
        if start is not None:
            out.append({"date": date, "start": start, "end": _clock(day["time"][-1], block_minutes)})
    return pl.DataFrame(out, schema={"date": pl.Utf8, "start": pl.Utf8, "end": pl.Utf8})


def share_windows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """The run's shared windows: each project with a ``share:`` and a ``window:``.

    ``project``, ``from``, ``to``, ``share`` (its mode, fraction and daily band) and ``members``:
    the project and those that ``share_with`` it (POLOCALC_ENG_PRE / _POST as POLOCALC_ENG).
    """
    projects = config.get("projects") or {}
    out = []
    for name, p in projects.items():
        share, window = (p or {}).get("share"), (p or {}).get("window")
        if not share or not window:
            continue
        members = {name} | {n for n, q in projects.items() if (q or {}).get("share_with") == name}
        out.append({
            "project": name, "from": str(window[0]), "to": str(window[1]),
            "members": sorted({project_name(n) for n in members}), "share": share,
        })
    return out


def in_window(df: pl.DataFrame, window: dict[str, Any]) -> pl.DataFrame:
    """The rows of a table (with a ``date``) inside a shared window."""
    if df.is_empty() or "date" not in df.columns:
        return df
    return df.filter((pl.col("date") >= window["from"]) & (pl.col("date") <= window["to"]))


def share_days(schedule: pl.DataFrame, window: dict[str, Any]) -> pl.DataFrame:
    """Day by day inside a shared window: the project's flights (``mine``), the others', all,
    the project's share, and whether it is in its daily band (``yes``, ``NO``, ``—`` without one)."""
    days_ = in_window(schedule.filter(pl.col("telescope").is_not_null()), window)
    daily = window["share"].get("daily") or {}
    lo, hi = daily.get("min"), daily.get("max")
    rows = []
    for (date,), x in days_.group_by("date", maintain_order=True):
        mine = int(x["project"].is_in(window["members"]).sum())
        frac = mine / len(x) if len(x) else 0.0
        ok = (lo is None or frac >= float(lo) - 1e-9) and (hi is None or frac <= float(hi) + 1e-9)
        rows.append({
            "date": date, "mine": mine, "others": len(x) - mine, "flights": len(x), "share": frac,
            "in_daily_band": ("yes" if ok else "NO") if daily else "—",
        })
    schema = {"date": pl.Utf8, "mine": pl.Int64, "others": pl.Int64, "flights": pl.Int64,
              "share": pl.Float64, "in_daily_band": pl.Utf8}
    return pl.DataFrame(rows, schema=schema).sort("date")


def flights_table(schedule: pl.DataFrame) -> pl.DataFrame:
    """The schedule's flights as the table shows them (:data:`FLIGHT_COLUMNS`, those it has).

    The drone's heading from its azimuth when the plan predates the column (toward the
    telescope: the azimuth + 180), ``band_crossed`` in percent.
    """
    if schedule.is_empty():
        return schedule
    df = schedule
    if "drone_heading_deg" not in df.columns and "drone_az" in df.columns:
        df = df.with_columns(
            ((pl.col("drone_az").cast(pl.Float64, strict=False) + 180) % 360).round(1).alias("drone_heading_deg")
        )
    if "band_crossed" in df.columns:
        df = df.with_columns(pl.col("band_crossed").cast(pl.Float64, strict=False) * 100)
    return df.select([c for c in FLIGHT_COLUMNS if c in df.columns])
