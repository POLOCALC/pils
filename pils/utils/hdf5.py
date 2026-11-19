"""Read HDF5 input files (the lab's raw and processed ``.h5py`` measurement files).

PILS saves its own results as parquet (:mod:`pils.store`); HDF5 is only an
input format here, written by the lab acquisition programs.
"""

from pathlib import Path
from typing import Any

import h5py


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
