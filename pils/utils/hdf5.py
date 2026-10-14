"""HDF5 helpers: nested reads and versioned analysis results.

Versioned results follow the layout of the flight analyses (PPK, EKF, AZEL)::

    <analysis>.h5
    └── [parent/]                    # optional, e.g. a telescope or a measurement
        └── rev_YYYYMMDD_HHMMSS/     # one group per run
            ├── [attrs] metadata     # JSON
            └── <table>/             # one group per DataFrame
                ├── <column>         # one dataset per column
                └── [attrs] columns, dtypes (JSON)
"""

import datetime
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pl

VERSION_PREFIX = "rev_"
VERSION_FORMAT = "rev_%Y%m%d_%H%M%S"


def h5_to_dict(obj: h5py.Group) -> dict[str, Any]:
    """Recursively read an HDF5 group into nested dicts of numpy arrays."""
    result: dict[str, Any] = {}
    for key, item in obj.items():
        if isinstance(item, h5py.Group):
            result[key] = h5_to_dict(item)
        else:
            result[key] = item[()]
    return result


def load_h5(path: str | Path) -> dict[str, Any]:
    """Read a whole HDF5 file into nested dicts of numpy arrays."""
    with h5py.File(path, "r") as f:
        return h5_to_dict(f)


def new_version_name(now: datetime.datetime | None = None) -> str:
    """Timestamped version name, e.g. ``rev_20260302_150712``."""
    return (now or datetime.datetime.now()).strftime(VERSION_FORMAT)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path | datetime.datetime | datetime.date):
        return str(value)
    raise TypeError(f"Not JSON serialisable: {type(value).__name__}")


def dumps_metadata(metadata: dict[str, Any]) -> str:
    """JSON for an HDF5 metadata attribute (numpy values allowed)."""
    return json.dumps(metadata, default=_json_default)


def write_table(parent: h5py.Group, name: str, frame: pl.DataFrame) -> None:
    """Store a DataFrame as a group with one dataset per column."""
    if name in parent:
        del parent[name]
    group = parent.create_group(name)
    for column in frame.columns:
        series = frame[column]
        if series.dtype == pl.Utf8:
            data = np.array(series.to_list(), dtype=h5py.string_dtype())
        else:
            data = series.to_numpy()
        group.create_dataset(column, data=data)
    group.attrs["columns"] = json.dumps(frame.columns)
    group.attrs["dtypes"] = json.dumps([str(d) for d in frame.dtypes])


def read_table(group: h5py.Group) -> pl.DataFrame:
    """Read a DataFrame written by :func:`write_table`."""
    columns = json.loads(group.attrs["columns"])
    dtypes = json.loads(group.attrs.get("dtypes", "[]")) or ["" for _ in columns]
    data: dict[str, Any] = {}
    for column, dtype in zip(columns, dtypes, strict=True):
        dataset = group[column]
        if dtype in ("String", "Utf8") or h5py.check_string_dtype(dataset.dtype):
            data[column] = list(dataset.asstr()[()])
        else:
            data[column] = dataset[()]
    return pl.DataFrame(data)


def _container(f: h5py.File, parent: str | None, create: bool) -> h5py.Group | None:
    if parent is None:
        return f
    if parent in f:
        return f[parent]
    return f.create_group(parent) if create else None


def save_version(
    path: str | Path,
    version_name: str,
    tables: dict[str, pl.DataFrame],
    metadata: dict[str, Any],
    parent: str | None = None,
) -> None:
    """Write one version (tables + metadata); replaces a version of the same name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        container = _container(f, parent, create=True)
        assert container is not None
        if version_name in container:
            del container[version_name]
        group = container.create_group(version_name)
        group.attrs["metadata"] = dumps_metadata(metadata)
        for name, frame in tables.items():
            write_table(group, name, frame)
        f.attrs["latest_version"] = version_name


def list_versions(path: str | Path, parent: str | None = None) -> list[str]:
    """Version names in a results file, oldest first (empty if no file)."""
    path = Path(path)
    if not path.is_file():
        return []
    with h5py.File(path, "r") as f:
        container = _container(f, parent, create=False)
        if container is None:
            return []
        return sorted(k for k in container.keys() if k.startswith(VERSION_PREFIX))


def load_version(
    path: str | Path, version_name: str, parent: str | None = None
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    """Read one version: (tables by name, metadata).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    KeyError
        If the version is not in the file.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {path}")
    with h5py.File(path, "r") as f:
        container = _container(f, parent, create=False)
        if container is None or version_name not in container:
            raise KeyError(f"Version '{version_name}' not found in {path}")
        group = container[version_name]
        metadata = json.loads(group.attrs.get("metadata", "{}"))
        tables = {
            name: read_table(item)
            for name, item in group.items()
            if isinstance(item, h5py.Group) and "columns" in item.attrs
        }
    return tables, metadata
