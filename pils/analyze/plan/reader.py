"""A planner's run folder: its tables and the configuration it ran with."""

from __future__ import annotations

import json
import re
from functools import cached_property
from pathlib import Path
from typing import Any

import polars as pl

# Tables with one row per key, whose counts add up when POLOCALC_ENG_PRE and _POST become one
# project: {file: the columns summed}; every other column is the key. The tables with a row per
# flight (the schedule, the joins, ...) keep their rows.
SUMMED: dict[str, tuple[str, ...]] = {
    "campaign_summary.csv": ("n_flights", "hours", "block_hours"),
    "detector_counts.csv": ("n_flights", "n_crossings"),
    "detector_counts_secondary.csv": ("n_crossings",),
    "detector_rotations.csv": ("n_flights",),
}


def project_name(name: Any) -> Any:
    """The project as the reader knows it: POLOCALC_ENG_PRE / _POST are POLOCALC_ENG."""
    return re.sub(r"_(PRE|POST)$", "", name) if isinstance(name, str) else name


def merge_projects(df: pl.DataFrame, name: str = "") -> pl.DataFrame:
    """``project`` with POLOCALC_ENG_PRE / _POST as one project; a table of counts (``SUMMED``)
    adds up the rows of a key, so that the key stays unique."""
    if "project" not in df.columns or df.is_empty():
        return df
    merged = df["project"].str.replace(r"_(PRE|POST)$", "")
    if merged.equals(df["project"]):
        return df
    df = df.with_columns(merged.alias("project"))
    summed = [c for c in SUMMED.get(name, ()) if c in df.columns]
    if not summed:
        return df
    keys = [c for c in df.columns if c not in summed]
    return (
        df.with_row_index("_order")
        .group_by(keys, maintain_order=True)
        .agg(pl.col("_order").first(), *[pl.col(c).sum() for c in summed])
        .sort("_order")
        .select(df.columns)
    )


def read_csv(path: str | Path) -> pl.DataFrame:
    """A planner's CSV: every column's type from all its rows, empty cells null (no file: empty)."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        return pl.DataFrame()
    return pl.read_csv(path, infer_schema_length=None, null_values=[""])


class PlanRun:
    """One run of the planner: ``<run>/plan/`` (or the plan folder itself)."""

    def __init__(self, path: str | Path, merge: bool = True):
        path = Path(path)
        self.path = path / "plan" if (path / "plan").is_dir() else path
        self.merge = merge

    @property
    def name(self) -> str:
        return self.path.parent.name if self.path.name == "plan" else self.path.name

    def has(self, name: str) -> bool:
        return (self.path / name).is_file()

    def table(self, name: str) -> pl.DataFrame:
        """A table of the plan (``campaign_schedule.csv``, ...); empty when it has none."""
        df = read_csv(self.path / name)
        return merge_projects(df, name) if self.merge else df

    @cached_property
    def meta(self) -> dict[str, Any]:
        """The run's record (``plan/run.json``, or ``config/run_plan.json`` beside it)."""
        for p in (self.path / "run.json", self.path.parent / "config" / "run_plan.json"):
            try:
                return json.loads(p.read_text())
            except (OSError, ValueError):
                continue
        return {}

    @cached_property
    def config(self) -> dict[str, Any]:
        """The configuration the run used, as saved beside it ({} if none is found or it is not YAML)."""
        import yaml

        name = Path(str(self.meta.get("config") or "campaign.yaml")).name
        config_dir = self.path.parent / "config"
        candidates = [self.path / name, config_dir / name, self.path / "campaign.yaml"]
        for p in candidates:
            if p.is_file():
                try:
                    return yaml.safe_load(p.read_text()) or {}
                except yaml.YAMLError:
                    return {}
        return {}
