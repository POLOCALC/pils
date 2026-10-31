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


# A plan in a campaign's folder (plans/<run>/): its tables a pils revision of proc/plan
PLAN_DATASET = "plan"
# The schedule's table in a revision (the CSV's other tables keep their names, snake case)
RENAMED = {"campaign_schedule": "schedule"}


def table_name(file: str) -> str:
    """A run's CSV as a table of the plan's revision: ``impact_LAT.csv`` -> ``impact_lat``."""
    stem = re.sub(r"\.csv$", "", file)
    stem = re.sub(r"[^a-z0-9_]+", "_", stem.lower()).strip("_")
    return RENAMED.get(stem, stem)


class PlanRun:
    """One plan of the planner: the plan in a campaign's folder (``plans/<run>/``, its tables a
    pils revision of ``proc/plan``), or a planner's run folder (``<run>/plan/*.csv``). Its tables
    are asked for by the run's file names (``campaign_schedule.csv``) either way."""

    def __init__(self, path: str | Path, merge: bool = True, revision: str | None = None):
        path = Path(path)
        self.merge = merge
        self.revision: Path | None = None
        if path.parent.name == PLAN_DATASET and path.name.startswith("rev_"):  # a revision itself
            self.revision, path = path, path.parents[2]
        elif (path / "proc" / PLAN_DATASET).is_dir():
            from pils import store

            rev = revision or store.latest_version(path / "proc", PLAN_DATASET)
            self.revision = path / "proc" / PLAN_DATASET / rev if rev else None
        if self.revision is not None:
            self.root = path  # plans/<run>: config/, report.md, figures/ beside proc/
            self.path = path
        else:
            self.path = path / "plan" if (path / "plan").is_dir() else path
            self.root = self.path.parent if self.path.name == "plan" else self.path

    @property
    def published(self) -> bool:
        """A plan in a campaign's folder (pils data), not a planner's run folder."""
        return self.revision is not None

    @property
    def name(self) -> str:
        return self.root.name

    @cached_property
    def _manifest(self) -> dict[str, Any]:
        if self.revision is None:
            return {}
        try:
            return json.loads((self.revision / "manifest.json").read_text())
        except (OSError, ValueError):
            return {}

    def names(self) -> list[str]:
        """The plan's tables, by the run's file names (``campaign_schedule.csv``, ...)."""
        if self.revision is None:
            return sorted(p.name for p in self.path.glob("*.csv"))
        files = (self._manifest.get("metadata") or {}).get("files") or {}
        known = {table: file for file, table in files.items()}
        back = {v: k for k, v in RENAMED.items()}
        return sorted(known.get(t, f"{back.get(t, t)}.csv") for t in self._manifest.get("tables", {}))

    def has(self, name: str) -> bool:
        if self.revision is None:
            return (self.path / name).is_file()
        return table_name(name) in self._manifest.get("tables", {})

    def table(self, name: str) -> pl.DataFrame:
        """A table of the plan (``campaign_schedule.csv``, ...); empty when it has none."""
        if self.revision is None:
            df = read_csv(self.path / name)
        elif self.has(name):
            df = pl.read_parquet(self.revision / f"{table_name(name)}.parquet")
            # as a CSV reads: an empty text is no value, a column of True / False booleans
            df = df.with_columns(
                pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c)
                for c, t in df.schema.items()
                if t == pl.Utf8
            )
            df = df.with_columns(
                (pl.col(c) == "True").alias(c)
                for c, t in df.schema.items()
                if t == pl.Utf8 and len(df) and set(df[c].drop_nulls().unique()) <= {"True", "False"}
                and df[c].drop_nulls().len()
            )
        else:
            df = pl.DataFrame()
        return merge_projects(df, name) if self.merge else df

    @cached_property
    def meta(self) -> dict[str, Any]:
        """The run's record (the revision's ``run_meta``; ``plan/run.json``, or
        ``config/run_plan.json`` beside it)."""
        if self.revision is not None:
            meta = (self._manifest.get("metadata") or {}).get("run_meta")
            if meta:
                return meta
        for p in (self.path / "run.json", self.root / "config" / "run_plan.json"):
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
        config_dir = self.root / "config"
        candidates = [self.path / name, config_dir / name, self.path / "campaign.yaml"]
        for p in candidates:
            if p.is_file():
                try:
                    loaded = yaml.safe_load(p.read_text()) or {}
                except yaml.YAMLError:
                    continue
                # a field file (fields: only) can carry the configuration's name: not it
                if isinstance(loaded, dict) and set(loaded) - {"fields"}:
                    return loaded
        return {}
