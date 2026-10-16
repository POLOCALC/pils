"""HDF5 helpers: nested reads and versioned analysis results.

Versioned results follow the layout of the flight analyses (PPK, EKF, AZEL)::

    <analysis>.h5
    └── [parent/]                    # optional, e.g. a telescope or a measurement
        └── rev_YYYYMMDD_HHMMSS/     # one group per run
            ├── [attrs] metadata     # JSON
            └── <table>/             # one group per DataFrame
                ├── <column>         # one dataset per column
                ├── _nulls/<column>  # null mask, for columns with nulls
                └── [attrs] columns, dtypes, units (JSON), n_rows, format
"""

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

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


# Version of the table layout written by :func:`write_table`
TABLE_FORMAT = 2
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


def write_table(
    parent: h5py.Group,
    name: str,
    frame: pl.DataFrame,
    units: dict[str, str] | None = None,
) -> None:
    """Store a DataFrame as a group with one dataset per column.

    - numbers and booleans as they are; dates and times as their integer
      physical values; text as UTF-8 strings;
    - nulls as a boolean mask per column in the ``_nulls`` subgroup (the
      stored value is 0, NaN or ""), so they come back as nulls;
    - gzip-compressed;
    - attributes: ``columns`` and ``dtypes`` (JSON), ``n_rows``,
      ``units`` (JSON, see :func:`pils.utils.column_units.table_units`) and
      ``format``.

    Columns of other types (lists, structs, objects) are skipped with a
    warning.

    Parameters
    ----------
    parent : h5py.Group
        Group to write into; an existing ``name`` is replaced.
    name : str
        Name of the table group.
    frame : pl.DataFrame
        Table to store.
    units : dict[str, str], optional
        Units by column; by default the known units of the columns.
    """
    from pils.utils.column_units import table_units

    if name in parent:
        del parent[name]
    group = parent.create_group(name)
    columns, dtypes = [], []
    for column in frame.columns:
        series = frame[column]
        dtype = series.dtype
        if dtype == pl.String:
            data = np.array(series.fill_null("").to_list(), dtype=h5py.string_dtype())
        elif dtype.is_temporal():
            data = series.to_physical().fill_null(0).to_numpy()
        elif dtype.is_float():
            data = series.fill_null(float("nan")).to_numpy()
        elif dtype.is_integer():
            data = series.fill_null(0).to_numpy()
        elif dtype == pl.Boolean:
            data = series.fill_null(False).to_numpy()
        else:
            logger.warning(f"{name}: column {column} ({dtype}) not stored in HDF5")
            continue
        compression = {"compression": "gzip"} if len(data) else {}
        group.create_dataset(column, data=data, **compression)
        if series.null_count():
            nulls = group.require_group(_NULLS)
            nulls.create_dataset(
                column, data=series.is_null().to_numpy(), compression="gzip"
            )
        columns.append(column)
        dtypes.append(str(dtype))
    group.attrs["columns"] = json.dumps(columns)
    group.attrs["dtypes"] = json.dumps(dtypes)
    group.attrs["n_rows"] = frame.height
    group.attrs["units"] = json.dumps(
        units if units is not None else table_units(columns)
    )
    group.attrs["format"] = TABLE_FORMAT


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


def read_units(group: h5py.Group) -> dict[str, str]:
    """Units stored with a table (empty for tables written without them)."""
    return _json_attr(group, "units", {})


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
