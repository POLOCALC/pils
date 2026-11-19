"""Writer of the old HDF5 results layout, to test ``pils migrate`` (PILS itself writes no HDF5)."""

import datetime
import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

TABLE_FORMAT = 2
_NULLS = "_nulls"


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
      ``units`` (JSON, see :func:`pils.schema.unit_of`) and
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
    from pils.schema import table_units

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
