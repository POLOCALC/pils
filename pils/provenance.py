"""Where a saved revision stands, and two revisions' data side by side.

The pils that made a revision against the one running, and two revisions compared.

Each revision's manifest says which pils made it (:func:`pils.store.code_version`: its version,
git commit, uncommitted changes). :func:`code_since` says how far that commit is behind the pils
running, and which of its files changed since -- to judge whether a flight is worth packaging
again. :func:`compare_revisions` compares two revisions of a dataset table by table and column by
column: rows, columns added or removed, and per numeric column its mean and spread in each, and the
largest difference sample by sample when both have the same times.

    code_since("eb1a54b...") -> {"behind": 12, "files": ["pils/synchronizer.py", ...], ...}
    compare_revisions(rev_a, rev_b) -> {"tables": [...]}
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils import store

SOURCE = Path(__file__).resolve().parent.parent
# The files of pils whose change can change a flight's data (not its tests, docs...)
CODE = "pils"
# The most changed files listed
MAX_FILES = 50
TIME_COLUMNS = ("timestamp", "time", "t")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SOURCE), *args],
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    ).stdout


@lru_cache(maxsize=256)
def _since(commit: str, head: str) -> dict[str, Any]:
    try:
        _git("cat-file", "-e", f"{commit}^{{commit}}")
    except (OSError, subprocess.SubprocessError):
        return {"known": False}
    try:
        _git("merge-base", "--is-ancestor", commit, head)
        ancestor = True
    except subprocess.CalledProcessError:
        ancestor = False
    behind = int(_git("rev-list", "--count", f"{commit}..{head}").strip() or 0)
    files = [
        f
        for f in _git("diff", "--name-only", commit, head, "--", CODE).splitlines()
        if f
    ]
    return {
        "known": True,
        "ancestor": ancestor,
        "behind": behind,
        "files": files[:MAX_FILES],
        "more_files": max(len(files) - MAX_FILES, 0),
    }


def code_since(git_commit: str | None) -> dict[str, Any]:
    """Describe a revision's pils commit against the pils running.

    Returns ``current`` (the running commit), ``same``, and when the commit is known to the
    running pils' git: ``ancestor`` (the running one comes after it), ``behind`` (commits since),
    ``files`` (of :data:`CODE` changed since, at most :data:`MAX_FILES`) and ``more_files``.
    ``known`` is False for a commit this git does not have, or a pils not run from a checkout.
    """
    running = store.code_version()
    head = running.get("git_commit")
    out: dict[str, Any] = {
        "current": head,
        "current_version": running.get("version"),
        "current_dirty": running.get("git_dirty"),
        "commit": git_commit,
        "same": bool(head and git_commit and head == git_commit),
    }
    if not head or not git_commit:
        return {**out, "known": False}
    if out["same"]:
        return {
            **out,
            "known": True,
            "ancestor": True,
            "behind": 0,
            "files": [],
            "more_files": 0,
        }
    return {**out, **_since(git_commit, head)}


def _time_column(frame: pl.DataFrame) -> str | None:
    return next((c for c in TIME_COLUMNS if c in frame.columns), None)


def _numeric(frame: pl.DataFrame) -> list[str]:
    return [c for c, t in frame.schema.items() if t.is_numeric() and t != pl.Boolean]


def _column(
    a: pl.Series | None, b: pl.Series | None, same_times: bool
) -> dict[str, Any]:
    def stats(s: pl.Series | None) -> tuple[float | None, float | None]:
        if s is None:
            return None, None
        x = s.cast(pl.Float64).to_numpy()
        x = x[np.isfinite(x)]
        return (float(x.mean()), float(x.std())) if x.size else (None, None)

    mean_a, std_a = stats(a)
    mean_b, std_b = stats(b)
    out: dict[str, Any] = {
        "mean": [mean_a, mean_b],
        "std": [std_a, std_b],
        "max_diff": None,
    }
    if same_times and a is not None and b is not None:
        d = a.cast(pl.Float64).to_numpy() - b.cast(pl.Float64).to_numpy()
        d = d[np.isfinite(d)]
        out["max_diff"] = float(np.abs(d).max()) if d.size else None
    return out


def compare_revisions(
    a: str | Path, b: str | Path, max_columns: int = 200
) -> dict[str, Any]:
    """Compare two revisions of a dataset, table by table (``a`` the newer, usually).

    Returns ``tables``: per table, ``status`` (same, changed, added -- in ``a`` only --,
    removed), ``rows`` [a, b], ``added`` and ``removed`` columns, ``same_times`` (both have the
    same time column values), and ``columns``: per numeric column in either, ``mean`` and ``std``
    [a, b] and ``max_diff`` (sample by sample, when the times are the same); ``changed`` the
    columns that differ. And ``pils`` [a, b], the pils that made each.
    """
    a, b = Path(a), Path(b)
    ma, mb = store.read_manifest(a), store.read_manifest(b)
    ta, tb = ma.get("tables") or {}, mb.get("tables") or {}
    tables = []
    for name in sorted(set(ta) | set(tb)):
        if name not in tb:
            tables.append(
                {"name": name, "status": "added", "rows": [ta[name].get("rows"), None]}
            )
            continue
        if name not in ta:
            tables.append(
                {
                    "name": name,
                    "status": "removed",
                    "rows": [None, tb[name].get("rows")],
                }
            )
            continue
        fa = store.read_parquet(store.table_path(a, name))
        fb = store.read_parquet(store.table_path(b, name))
        time_a, time_b = _time_column(fa), _time_column(fb)
        same_times = bool(
            time_a
            and time_a == time_b
            and fa.height == fb.height
            and np.array_equal(
                fa[time_a].to_numpy(), fb[time_b].to_numpy(), equal_nan=True
            )
        )
        numeric = sorted(set(_numeric(fa)) | set(_numeric(fb)))[:max_columns]
        columns = {
            c: _column(
                fa[c] if c in fa.columns else None,
                fb[c] if c in fb.columns else None,
                same_times,
            )
            for c in numeric
        }

        def differs(c: str, s: dict[str, Any]) -> bool:
            if s["max_diff"] is not None:
                return s["max_diff"] > 0
            return s["mean"][0] != s["mean"][1] or s["std"][0] != s["std"][1]

        changed = [c for c, s in columns.items() if differs(c, s)]
        added = sorted(set(fa.columns) - set(fb.columns))
        removed = sorted(set(fb.columns) - set(fa.columns))
        tables.append(
            {
                "name": name,
                "status": "changed"
                if changed or added or removed or fa.height != fb.height
                else "same",
                "rows": [fa.height, fb.height],
                "added": added,
                "removed": removed,
                "same_times": same_times,
                "time_column": time_a if time_a == time_b else None,
                "columns": columns,
                "changed": changed,
            }
        )
    return {"tables": tables, "pils": [ma.get("pils"), mb.get("pils")]}
