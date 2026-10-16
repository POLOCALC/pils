"""Write-once results: one directory of Parquet tables per revision.

Every product (the flight's raw and synchronized data, AZEL, power, EKF,
PPK) is stored the same way::

    <root>/[<parent>/]rev_YYYYMMDD_hhmmss/
    ├── manifest.json        # what, when, with which code, from which inputs
    ├── <table>.parquet      # one file per table; "/" in a name is a folder
    └── ...                  # other files of the run (plots, CSV)

- A revision is complete when its ``manifest.json`` exists; it is written
  last, after the tables, and is never overwritten: a revision does not
  change once saved.
- Parquet keeps the column types, nulls, time zones and the units (in the
  file's ``pils`` metadata, see :mod:`pils.schema`); polars, pandas, pyarrow,
  DuckDB, R, Julia and MATLAB read it without pils.
- The manifest records the schema version, the code (pils version, git
  commit and whether the tree had changes), the tables (rows, columns,
  dtypes, units) and the run's own metadata.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import subprocess
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from pils import schema

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
VERSION_PREFIX = "rev_"
VERSION_FORMAT = "rev_%Y%m%d_%H%M%S"

Tables = dict[str, "pl.DataFrame | Tables"]


def new_version_name(now: datetime.datetime | None = None) -> str:
    """Timestamped version name, e.g. ``rev_20260302_150712``."""
    return (now or datetime.datetime.now()).strftime(VERSION_FORMAT)


@lru_cache(maxsize=1)
def code_version() -> dict[str, Any]:
    """The running pils: package version, git commit, uncommitted changes."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        package = version("pils")
    except PackageNotFoundError:
        package = "unknown"
    info: dict[str, Any] = {"version": package, "git_commit": None, "git_dirty": None}
    source = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--", "pils"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        info["git_commit"] = commit
        info["git_dirty"] = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        pass  # an installed copy, not a git checkout
    return info


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, datetime.datetime | datetime.date | datetime.time):
        return value.isoformat()
    if isinstance(value, Path | datetime.timedelta):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    raise TypeError(f"Not JSON serialisable: {type(value).__name__}")


def dumps(data: Any) -> str:
    """JSON with numpy values, paths and datetimes allowed."""
    return json.dumps(data, default=_json_default, indent=2)


# -- tables -------------------------------------------------------------


def write_parquet(frame: pl.DataFrame, path: str | Path) -> dict[str, Any]:
    """Write a table; the units of its columns go in the file metadata.

    Returns
    -------
    dict
        Rows, and dtype and unit of each column, for the manifest.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    units = schema.table_units(frame.columns)
    table = frame.to_arrow()
    meta = dict(table.schema.metadata or {})
    meta[b"pils"] = json.dumps(
        {"schema_version": schema.SCHEMA_VERSION, "units": units}
    ).encode()
    pq.write_table(table.replace_schema_metadata(meta), path, compression="zstd")
    return {
        "rows": frame.height,
        "columns": {
            c: {"dtype": str(t), **({"unit": units[c]} if c in units else {})}
            for c, t in frame.schema.items()
        },
    }


def read_parquet(path: str | Path) -> pl.DataFrame:
    """Read a table written by :func:`write_parquet` (or any Parquet file)."""
    return pl.read_parquet(path)


def read_units(path: str | Path) -> dict[str, str]:
    """Units stored in a Parquet file written by :func:`write_parquet`."""
    meta = pq.read_schema(path).metadata or {}
    if b"pils" not in meta:
        return {}
    return json.loads(meta[b"pils"])["units"]


def _flatten(tables: Tables, prefix: str = "") -> dict[str, pl.DataFrame]:
    flat: dict[str, pl.DataFrame] = {}
    for name, value in tables.items():
        key = f"{prefix}{name}"
        if isinstance(value, pl.DataFrame):
            flat[key] = value
        elif isinstance(value, dict):
            flat.update(_flatten(value, f"{key}/"))
        elif value is not None:
            raise TypeError(f"Table {key} is a {type(value).__name__}")
    return flat


def _nest(flat: dict[str, pl.DataFrame]) -> Tables:
    tables: Tables = {}
    for key, frame in flat.items():
        node = tables
        *parents, name = key.split("/")
        for parent in parents:
            node = node.setdefault(parent, {})  # type: ignore[assignment]
        node[name] = frame
    return tables


# -- revisions ----------------------------------------------------------


def _container(root: str | Path, parent: str | None) -> Path:
    return Path(root) / parent if parent else Path(root)


def write_tables(
    directory: str | Path,
    tables: Tables,
    metadata: dict[str, Any] | None = None,
    kind: str | None = None,
    replace: bool = False,
) -> Path:
    """Write tables and their manifest into ``directory``.

    The tables go to a temporary folder first. Without ``replace`` the
    directory may already hold other files of the run (plots, CSV) but not a
    manifest; with ``replace`` the whole directory is swapped for the new one
    (for caches such as the flight's raw data).

    Parameters
    ----------
    directory : str or Path
        Destination.
    tables : dict
        Tables by name; nested dicts become folders.
    metadata : dict, optional
        Stored in the manifest under ``metadata``.
    kind : str, optional
        What the tables are, e.g. ``"flight_sync"`` or ``"azel"``.
    replace : bool, default=False
        Replace an existing directory.

    Returns
    -------
    Path
        The directory.

    Raises
    ------
    FileExistsError
        If the directory already has a manifest and ``replace`` is False.
    """
    directory = Path(directory)
    if not replace and (directory / MANIFEST).exists():
        raise FileExistsError(f"{directory} is already saved and is not changed")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.{uuid.uuid4().hex[:8]}.tmp"
    staging.mkdir()
    try:
        entries = {
            key: write_parquet(frame, staging / f"{key}.parquet")
            for key, frame in _flatten(tables).items()
        }
        manifest = {
            "schema_version": schema.SCHEMA_VERSION,
            "kind": kind,
            "saved_at": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            ),
            "pils": code_version(),
            "tables": entries,
            "metadata": metadata or {},
        }
        if replace or not directory.exists():
            (staging / MANIFEST).write_text(dumps(manifest))
            if directory.exists():
                old = directory.parent / f".{directory.name}.{uuid.uuid4().hex[:8]}.old"
                os.rename(directory, old)
                os.rename(staging, directory)
                shutil.rmtree(old)
            else:
                os.rename(staging, directory)
        else:
            # Next to the run's other files: tables first, manifest last
            for item in staging.iterdir():
                target = directory / item.name
                if target.exists():
                    raise FileExistsError(f"{target} exists")
                os.rename(item, target)
            (directory / MANIFEST).write_text(dumps(manifest))
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return directory


def read_tables(directory: str | Path) -> tuple[Tables, dict[str, Any]]:
    """Tables (nested by folder) and manifest of a saved directory."""
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST).read_text())
    flat = {
        key: read_parquet(directory / f"{key}.parquet") for key in manifest["tables"]
    }
    return _nest(flat), manifest


def save_version(
    root: str | Path,
    version_name: str,
    tables: Tables,
    metadata: dict[str, Any],
    parent: str | None = None,
    kind: str | None = None,
) -> Path:
    """Save one revision: ``<root>/[<parent>/]<version_name>/``."""
    return write_tables(
        _container(root, parent) / version_name, tables, metadata, kind=kind
    )


def list_versions(root: str | Path, parent: str | None = None) -> list[str]:
    """Saved revisions (with a manifest), oldest first."""
    container = _container(root, parent)
    if not container.is_dir():
        return []
    return sorted(
        item.name
        for item in container.iterdir()
        if item.name.startswith(VERSION_PREFIX) and (item / MANIFEST).is_file()
    )


def load_version(
    root: str | Path, version_name: str, parent: str | None = None
) -> tuple[Tables, dict[str, Any]]:
    """Tables and run metadata of one revision.

    Raises
    ------
    KeyError
        If the revision is not saved.
    """
    directory = _container(root, parent) / version_name
    if not (directory / MANIFEST).is_file():
        raise KeyError(f"Version '{version_name}' not found in {directory.parent}")
    tables, manifest = read_tables(directory)
    return tables, manifest["metadata"]


def read_manifest(directory: str | Path) -> dict[str, Any]:
    """The manifest of a saved directory."""
    return json.loads((Path(directory) / MANIFEST).read_text())


def delete_version(
    root: str | Path, version_name: str, parent: str | None = None
) -> None:
    """Remove a revision and its files."""
    directory = _container(root, parent) / version_name
    if directory.is_dir():
        shutil.rmtree(directory)
