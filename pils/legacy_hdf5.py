"""Read-only reader of the HDF5 results PILS saved before :mod:`pils.store`.

Only ``pils migrate`` uses it, to convert the old ``<analysis>_solution.h5``
files to parquet. PILS writes no HDF5; delete this module (and
:mod:`pils.migrate`) once the old files are converted.

Layout of a results file::

    <analysis>_solution.h5
    └── [parent/]                    # optional, e.g. a telescope or a measurement
        └── rev_YYYYMMDD_HHMMSS/     # one group per run
            ├── [attrs] metadata     # JSON
            └── <table>/             # one group per DataFrame
                ├── <column>         # one dataset per column
                ├── _nulls/<column>  # null mask, for columns with nulls
                └── [attrs] columns, dtypes, units (JSON), n_rows, format
"""

import json
import re
from pathlib import Path
from typing import Any

import h5py
import polars as pl

from pils import store

VERSION_PREFIX = store.VERSION_PREFIX
_NULLS = "_nulls"


def parse_dtype(dtype: str) -> pl.DataType | None:
    """Polars dtype from its string form (``str(dtype)``), or None.

    Examples
    --------
    >>> parse_dtype("Datetime(time_unit='us', time_zone='UTC')")
    Datetime(time_unit='us', time_zone='UTC')
    """
    simple = {
        "Float64": pl.Float64,
        "Float32": pl.Float32,
        "Int64": pl.Int64,
        "Int32": pl.Int32,
        "Int16": pl.Int16,
        "Int8": pl.Int8,
        "UInt64": pl.UInt64,
        "UInt32": pl.UInt32,
        "UInt16": pl.UInt16,
        "UInt8": pl.UInt8,
        "Boolean": pl.Boolean,
        "String": pl.String,
        "Utf8": pl.String,
        "Date": pl.Date,
        "Time": pl.Time,
    }
    if dtype in simple:
        return simple[dtype]
    match = re.fullmatch(
        r"(Datetime|Duration)\(time_unit='(\w+)'(?:, time_zone=(?:'([^']*)'|None))?\)",
        dtype,
    )
    if match:
        kind, unit, zone = match.groups()
        if kind == "Datetime":
            return pl.Datetime(unit, zone)  # type: ignore[arg-type]
        return pl.Duration(unit)  # type: ignore[arg-type]
    return None


def _json_attr(group: h5py.Group, name: str, default: Any) -> Any:
    value = group.attrs.get(name)
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode()
    return json.loads(str(value))


def read_table(group: h5py.Group, legacy_time_unit: str | None = None) -> pl.DataFrame:
    """Read a DataFrame written by :func:`write_table`.

    Parameters
    ----------
    group : h5py.Group
        Table group.
    legacy_time_unit : str, optional
        For tables written before the ``format`` attribute: the unit of the
        stored datetimes, if not their own (PPK stored microseconds).

    Returns
    -------
    pl.DataFrame
        The table, with its types, nulls and column order.
    """
    columns = _json_attr(group, "columns", [])
    dtypes = _json_attr(group, "dtypes", []) or ["" for _ in columns]
    legacy = "format" not in group.attrs
    nulls = group.get(_NULLS)
    series = []
    for column, dtype_name in zip(columns, dtypes, strict=False):
        if column not in group:
            continue
        dataset = group[column]
        dtype = parse_dtype(dtype_name)
        if h5py.check_string_dtype(dataset.dtype):
            values = pl.Series(column, list(dataset.asstr()[()]), dtype=pl.String)
        else:
            values = pl.Series(column, dataset[()])
        if dtype is not None and dtype.is_temporal():
            if (
                legacy
                and legacy_time_unit is not None
                and isinstance(dtype, pl.Datetime)
            ):
                values = values.cast(pl.Datetime(legacy_time_unit, dtype.time_zone))  # type: ignore[arg-type]
                values = values.dt.cast_time_unit(dtype.time_unit)  # type: ignore[arg-type]
            else:
                values = values.cast(dtype)
        elif dtype is not None and dtype != values.dtype:
            values = values.cast(dtype, strict=False)
        if nulls is not None and column in nulls:
            mask = pl.Series(nulls[column][()])
            values = (
                pl.select(pl.when(pl.lit(mask)).then(None).otherwise(pl.lit(values)))
                .to_series()
                .alias(column)
            )
        series.append(values)
    return pl.DataFrame(series)


def _container(f: h5py.File, parent: str | None) -> h5py.Group | None:
    if parent is None:
        return f
    return f[parent] if parent in f else None


def list_versions(path: str | Path, parent: str | None = None) -> list[str]:
    """Version names in a results file, oldest first (empty if no file)."""
    path = Path(path)
    if not path.is_file():
        return []
    with h5py.File(path, "r") as f:
        container = _container(f, parent)
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
        container = _container(f, parent)
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
