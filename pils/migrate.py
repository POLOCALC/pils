"""Convert results saved in HDF5 (before pils.store) to the store layout.

The HDF5 files are kept; revisions already converted are skipped. Column
names are converted to the schema (:mod:`pils.schema`).

=====================  =====================================================
HDF5 file              becomes
=====================  =====================================================
``sync_data.h5``       ``flight/raw/`` and ``flight/sync/<rev>/``
``azel_solution.h5``   ``azel/<TELESCOPE>/<rev>/``
``power_solution.h5``  ``power/<TELESCOPE>_<wafer slot>/<rev>/``
``ekf_solution.h5``    ``ekf/<rev>/``
``ppk_solution.h5``    ``ppk/<rev>/``
=====================  =====================================================
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py

from pils import store
from pils.utils.hdf5 import read_table

logger = logging.getLogger(__name__)

# AZEL tables before the schema
_AZEL_NAMES = {
    "az": "az_deg",
    "el": "el_deg",
    "srange": "srange_m",
    "E": "e_m",
    "N": "n_m",
    "U": "u_m",
}


def _attrs(group: h5py.Group) -> dict[str, Any]:
    """Attributes as JSON values (JSON strings are parsed)."""
    out: dict[str, Any] = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        if hasattr(value, "item"):
            value = value.item() if getattr(value, "ndim", 1) == 0 else value.tolist()
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError:
                pass
        out[key] = value
    return out


def _versions(
    path: Path, root: Path, parent_level: bool, kind: str, table: str, dry_run: bool
) -> list[str]:
    """Versions stored as ``[<parent>/]<rev>/<table>`` with JSON metadata."""
    done = []
    with h5py.File(path, "r") as f:
        containers = (
            [(name, f[name]) for name in f if isinstance(f[name], h5py.Group)]
            if parent_level
            else [(None, f)]
        )
        for parent, container in containers:
            for name, group in container.items():
                if not name.startswith(store.VERSION_PREFIX) or table not in group:
                    continue
                if name in store.list_versions(root, parent):
                    continue
                label = f"{kind} {parent + '/' if parent else ''}{name}"
                done.append(label)
                if dry_run:
                    continue
                frame = read_table(group[table], legacy_time_unit="us")
                if kind == "azel":
                    frame = frame.rename(
                        {k: v for k, v in _AZEL_NAMES.items() if k in frame.columns}
                    )
                attrs = _attrs(group)
                metadata = attrs.pop("metadata", None) or attrs
                store.save_version(
                    root, name, {table: frame}, metadata, parent=parent, kind=kind
                )
    return done


def _ppk(path: Path, root: Path, dry_run: bool) -> list[str]:
    done = []
    with h5py.File(path, "r") as f:
        for name, group in f.items():
            if name in store.list_versions(root):
                continue
            done.append(f"ppk {name}")
            if dry_run:
                continue
            store.save_version(
                root,
                name,
                {
                    "position": read_table(group["position"], legacy_time_unit="us"),
                    "statistics": read_table(
                        group["statistics"], legacy_time_unit="us"
                    ),
                },
                _attrs(group),
                kind="ppk",
            )
    return done


def _flight(path: Path, root: Path, dry_run: bool) -> list[str]:
    from pils.flight import Flight

    with h5py.File(path, "r") as f:
        revisions = sorted(f.get("sync_data", {}))
        metadata = {r: _attrs(f["sync_data"][r]) for r in revisions}
    existing = store.list_versions(root / "sync")
    todo = [r for r in revisions if r not in existing]
    if dry_run:
        return [f"flight sync {r}" for r in todo]
    done = []
    if not (root / "raw" / store.MANIFEST).exists() and not todo:
        Flight.from_hdf5(path, sync_version=False).save(root)
        done.append("flight raw")
    for revision in todo:
        flight = Flight.from_hdf5(path, sync_version=revision)
        info = {
            k.removeprefix("user_"): v
            for k, v in metadata[revision].items()
            if k not in ("created_at", "n_keys")
        }
        flight.save(root, sync_metadata=info, revision=revision)
        done.append(f"flight sync {revision}")
    return done


def migrate_proc(proc_dir: str | Path, dry_run: bool = False) -> list[str]:
    """Convert the HDF5 results of a flight's ``proc/`` folder.

    Parameters
    ----------
    proc_dir : str or Path
        The flight's ``proc`` folder.
    dry_run : bool, default=False
        Only list what would be converted.

    Returns
    -------
    list[str]
        The revisions converted (or to convert).
    """
    proc = Path(proc_dir)
    done: list[str] = []
    jobs = [
        ("sync_data.h5", lambda p: _flight(p, proc / "flight", dry_run)),
        (
            "azel/azel_solution.h5",
            lambda p: _versions(p, proc / "azel", True, "azel", "azel_data", dry_run),
        ),
        (
            "power/power_solution.h5",
            lambda p: _versions(
                p, proc / "power", True, "power", "power_data", dry_run
            ),
        ),
        (
            "ekf/ekf_solution.h5",
            lambda p: _versions(p, proc / "ekf", False, "ekf", "ekf_data", dry_run),
        ),
        ("ppk/ppk_solution.h5", lambda p: _ppk(p, proc / "ppk", dry_run)),
    ]
    for relative, job in jobs:
        path = proc / relative
        if path.is_file():
            try:
                done += job(path)
            except Exception as e:
                logger.error(f"{path}: {type(e).__name__}: {e}")
                done.append(f"FAILED {relative}: {type(e).__name__}: {e}")
    return done
