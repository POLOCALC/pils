"""A field search (or a field file evaluated): what it found, shaped for its plots.

A planner's field search writes, in ``output/fields/<run>/optimized_fields[/<tag>]/``:
``candidates.csv`` (every grid point scored), ``fields.yaml`` (the fields kept, set from
a reference telescope, with their geometry from every telescope), ``windows.npz`` (per
field, telescope and wafer, per day and block: flyable, blocked by another telescope, the
rotations Sun-safe), ``conflicts_<TEL>.csv`` (how close each avoided telescope's scans
come to the fields) and ``report.txt``. :class:`FieldSearch` reads them; the functions
return plot-ready data.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils.analyze.plan.reader import read_csv


def searches(run_dir: str | Path) -> list[dict[str, str]]:
    """Every search of a field run: its folder (relative to the run) and its label ("tool / tag"). A
    folder that only holds tagged searches is not one itself unless it has its own run.json."""
    tool = Path(run_dir) / "optimized_fields"
    if not tool.is_dir():
        return []
    tags = sorted(p for p in tool.iterdir() if p.is_dir())
    out = [{"path": tool.name, "label": tool.name}] if (tool / "run.json").is_file() or not tags else []
    return out + [{"path": f"{tool.name}/{t.name}", "label": f"{tool.name} / {t.name}"} for t in tags]


class FieldSearch:
    """One search's folder."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @cached_property
    def meta(self) -> dict[str, Any]:
        try:
            return json.loads((self.path / "run.json").read_text())
        except (OSError, ValueError):
            return {}

    @property
    def argv(self) -> list[str]:
        return list(self.meta.get("argv") or [])

    @cached_property
    def candidates(self) -> pl.DataFrame:
        return read_csv(self.path / "candidates.csv")

    @cached_property
    def fields_text(self) -> str:
        p = self.path / "fields.yaml"
        return p.read_text() if p.is_file() else ""

    @cached_property
    def fields(self) -> dict[str, Any]:
        import yaml

        return ((yaml.safe_load(self.fields_text) or {}).get("fields") or {}) if self.fields_text else {}

    @cached_property
    def windows(self) -> dict[str, np.ndarray] | None:
        p = self.path / "windows.npz"
        return dict(np.load(p)) if p.is_file() else None

    def conflicts(self) -> dict[str, pl.DataFrame]:
        return {p.stem[len("conflicts_"):]: read_csv(p) for p in sorted(self.path.glob("conflicts_*.csv"))}

    @property
    def report(self) -> str:
        p = self.path / "report.txt"
        return p.read_text(errors="replace") if p.is_file() else ""

    def config_path(self) -> Path | None:
        """The configuration the search ran with: the copy it kept, else the file it names."""
        kept = sorted(p for p in self.path.glob("*.yaml") if p.name != "fields.yaml")
        named = Path(self.argv[1]).name if len(self.argv) > 1 else None
        if kept:
            return next((p for p in kept if p.name == named), kept[0])
        return None

    def drone_limits(self) -> dict[str, float]:
        """The drone limits the search was given (--min-distance-m, --ceiling-m, --min-el-deg)."""
        out = {}
        for flag, key in (("--min-distance-m", "min_range_m"), ("--ceiling-m", "max_altitude_m"),
                          ("--min-el-deg", "min_el_deg")):
            if flag in self.argv:
                out[key] = float(self.argv[self.argv.index(flag) + 1])
        return out


def chosen_table(fields: dict[str, Any]) -> list[dict[str, Any]]:
    """The fields kept: each with the telescope it is set from and, per telescope, "az / el · range m"."""
    return [{"field": n, "set from": (spec.get("reference") or {}).get("telescope"),
             **{t: f"{g['az']:.1f} / {g['el']:.1f} · {g.get('range_m', 0):.0f} m"
                for t, g in (spec.get("geometry") or {}).items()}}
            for n, spec in fields.items()]


def landscape(candidates: pl.DataFrame, metric: str = "usable_h_per_day") -> dict[str, Any]:
    """The candidates on the reference telescope's sky: a grid of azimuth x elevation, each cell the best
    of `metric` there (None where nothing was scored), and the best candidate."""
    if candidates.is_empty():
        return {"az": [], "el": [], "grid": [], "best": None}
    az_col = next(c for c in candidates.columns if c.startswith("az_from_"))
    el_col = next(c for c in candidates.columns if c.startswith("el_from_"))
    g = candidates.group_by(az_col, el_col).agg(pl.col(metric).max())
    azs, els = sorted(g[az_col].unique().to_list()), sorted(g[el_col].unique().to_list())
    cell = {(a, e): v for a, e, v in g.iter_rows()}
    best = candidates.row(int(candidates[metric].arg_max()), named=True)
    return {"reference": az_col[len("az_from_"):], "az": azs, "el": els,
            "grid": [[cell.get((a, e)) for a in azs] for e in els],
            "best": {"az": best[az_col], "el": best[el_col], "value": best[metric]}}


def _segments(mask: np.ndarray, hours: np.ndarray, step: float) -> list[list[float]]:
    """The runs of a boolean row as [start, end] hours."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return [[float(hours[a]), float(hours[z - 1] + step)] for a, z in zip(edges[::2], edges[1::2])]


def field_windows(win: dict[str, np.ndarray], field: str, telescope: str | None = None, wafer: str | None = None,
                  day: str | None = None, by_rotation: bool = False) -> dict[str, Any]:
    """When a field can be flown over the campaign (``windows.npz``): per lane -- a telescope (at least one
    wafer), a wafer of one telescope, or a boresight rotation -- and per day, the stretches flyable
    (``ok``: inside the envelope, Sun-safe at some rotation, no telescope disturbed), blocked by another
    telescope (``other``) or by the telescope's own High Priority scan (``own``), and the flyable hours per
    day (over the days the search scored). `day`: that day alone."""
    dates, times = [str(d) for d in win["dates"]], [str(t) for t in win["times"]]
    hours = np.array([int(t[:2]) + int(t[3:]) / 60 for t in times])
    step = float(hours[1] - hours[0]) if len(hours) > 1 else 0.5
    keys = [k.split("|") for k in win if k.count("|") == 3 and k.startswith(field + "|") and k.endswith("|flyable")]
    outside: list[str] = []
    if telescope is None:
        acc: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for _, t, w, _ in keys:
            ok, bl = win[f"{field}|{t}|{w}|flyable"], win[f"{field}|{t}|{w}|blocked"]
            a, b = acc.get(t, (np.zeros_like(ok), np.zeros_like(bl)))
            acc[t] = (a | ok, b | bl)
        lanes = {t: (ok, bl & ~ok, t) for t, (ok, bl) in acc.items()}
    else:
        mine = [w for _, t, w, _ in keys if t == telescope and (wafer is None or w == wafer)]
        lanes = {w: (win[f"{field}|{telescope}|{w}|flyable"], win[f"{field}|{telescope}|{w}|blocked"], telescope)
                 for w in mine}
        outside = [w for w in mine if not bool(win.get(f"{field}|{telescope}|{w}|inside", True))]
    rots = win.get(f"rotations|{telescope}")
    if by_rotation and telescope is not None and rots is not None and f"{field}|{telescope}|clear" in win:
        clear = win[f"{field}|{telescope}|clear"]
        mine = [w for _, t, w, _ in keys if t == telescope and (wafer is None or w == wafer)]
        lanes = {}
        for i, b in enumerate(rots):
            safe = np.zeros_like(clear)
            for w in mine:
                safe |= (win[f"{field}|{telescope}|{w}|rot"] >> i) & 1 > 0
            lanes[f"BS {int(b):+d}°"] = (safe & clear, safe & ~clear, telescope)
    rows = [dates.index(day)] if day in dates else list(range(len(dates)))
    scored = [i for i in rows if bool(win["days"][i])] if "days" in win else rows
    out_lanes = []
    for name, (ok, bl, tel) in lanes.items():
        own = win.get(f"{field}|{tel}|own")
        own = np.zeros_like(bl) if own is None else own
        out_lanes.append({
            "name": name,
            "hours_per_day": float(ok[rows].sum() * step / max(len(scored), 1)),
            "rows": [{"ok": _segments(ok[i], hours, step), "other": _segments((bl & ~own)[i], hours, step),
                      "own": _segments((bl & own)[i], hours, step)} for i in rows],
        })
    return {"dates": [dates[i] for i in rows], "start": float(hours[0]), "end": float(hours[-1] + step),
            "lanes": out_lanes, "outside": outside,
            "telescopes": sorted({t for _, t, _, _ in keys}),
            "wafers": sorted({w for _, t, w, _ in keys if t == telescope}) if telescope else [],
            "all_dates": dates, "has_rotations": telescope is not None and rots is not None}


def separation(conflicts: pl.DataFrame, fields: list[str], day: str | None = None) -> dict[str, Any]:
    """How close a telescope's scans come to each field over one day (its ``conflicts_<TEL>.csv``): per
    field, time and separation (with the scan and the minutes in conflict), and the exclusion angle."""
    df = conflicts.filter(pl.col("field").is_in(fields)) if fields else conflicts
    days = sorted(df["date"].unique().to_list()) if not df.is_empty() else []
    day = day if day in days else (days[0] if days else None)
    one = df.filter(pl.col("date") == day) if day else df.clear()
    series = []
    for (f,), g in one.group_by("field", maintain_order=True):
        series.append({"field": f, "time": g["time"].to_list(),
                       "separation_deg": g["separation_deg"].cast(pl.Float64, strict=False).to_list(),
                       "scan": g["scan"].to_list() if "scan" in g.columns else [],
                       "conflict_min": g["conflict_min"].to_list() if "conflict_min" in g.columns else []})
    excl = one["exclusion_deg"].drop_nulls() if "exclusion_deg" in one.columns and not one.is_empty() else []
    return {"days": days, "day": day, "series": series, "exclusion_deg": float(excl[0]) if len(excl) else None}


def sun_safe(times: list[str], separation: np.ndarray, threshold: float, day: int | None = None) -> list[list[float]]:
    """The Sun-safe stretches of one scan over a day ([start, end] hours), or -- `day` None -- the joint
    flag: safe only where it is safe on every day."""
    hours = np.array([int(t[:2]) + int(t[3:]) / 60 for t in times])
    step = float(hours[1] - hours[0]) if len(hours) > 1 else 1 / 30
    sep = np.asarray(separation, float)
    unsafe = (sep < threshold).any(axis=0) if day is None else sep[day] < threshold
    return _segments(~unsafe, hours, step)
