"""Convert results saved in HDF5 (before pils.store) to parquet (the store layout).

The only place PILS reads HDF5 results (see :mod:`pils.legacy_hdf5`): delete it
once the old files are converted. The HDF5 files are kept; revisions already
converted are skipped. Column names are converted to the schema
(:mod:`pils.schema`): AZEL, EKF and PPK
(:func:`pils.analyze.ppk.legacy_ppk_tables`). Flight files in the single-HDF5
format (``sync_data.h5``) are not converted any more.

=====================  =====================================================
HDF5 file              becomes
=====================  =====================================================
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
from pils.analyze.ekf import EKF_OUTPUT_NAMES
from pils.analyze.ppk import legacy_ppk_tables
from pils.legacy_hdf5 import read_table

logger = logging.getLogger(__name__)

# Flight results in <proc>/<analysis>/<analysis>_solution.h5 (not lab runs)
FLIGHT_ANALYSES = frozenset({"azel", "power", "ekf", "ppk"})

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
                names = {"azel": _AZEL_NAMES, "ekf": EKF_OUTPUT_NAMES}.get(kind, {})
                frame = frame.rename(
                    {k: v for k, v in names.items() if k in frame.columns}
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
            position, statistics = legacy_ppk_tables(
                read_table(group["position"], legacy_time_unit="us"),
                read_table(group["statistics"], legacy_time_unit="us"),
            )
            store.save_version(
                root,
                name,
                {"position": position, "statistics": statistics},
                _attrs(group),
                kind="ppk",
            )
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
            except Exception as e:  # Broad: a failed file is reported, the others go on
                logger.exception(f"{path}: {type(e).__name__}: {e}")
                done.append(f"FAILED {relative}: {type(e).__name__}: {e}")
    return done


def migrate_lab(proc_dir: str | Path, dry_run: bool = False) -> list[str]:
    """Convert the HDF5 lab results of a test folder's ``proc/`` folder.

    ``proc/<analysis>/<analysis>_solution.h5`` (one group per measurement
    stem, one per run) becomes ``proc/<analysis>/<stem>/<rev>/`` (see
    :mod:`pils.store`). The HDF5 files are kept; runs already converted are
    skipped. Lab tables already use the schema's names.

    Parameters
    ----------
    proc_dir : str or Path
        The test folder's ``proc`` folder, e.g.
        ``<lab root>/<band>/<category>/<item>/<test_type>/proc``.
    dry_run : bool, default=False
        Only list what would be converted.

    Returns
    -------
    list[str]
        The runs converted (or to convert), as ``<analysis> <stem>/<rev>``.
    """
    from pils import legacy_hdf5 as hdf5

    done: list[str] = []
    for path in sorted(Path(proc_dir).glob("*/*_solution.h5")):
        root = path.parent
        analysis = path.name.removesuffix("_solution.h5")
        if analysis in FLIGHT_ANALYSES:  # converted by migrate_proc
            continue
        try:
            with h5py.File(path, "r") as f:
                stems = [name for name in f if isinstance(f[name], h5py.Group)]
            for stem in stems:
                existing = store.list_versions(root, parent=stem)
                for name in hdf5.list_versions(path, parent=stem):
                    if name in existing:
                        continue
                    done.append(f"{analysis} {stem}/{name}")
                    if dry_run:
                        continue
                    tables, metadata = hdf5.load_version(path, name, parent=stem)
                    store.save_version(
                        root,
                        name,
                        dict(tables),
                        metadata,
                        parent=stem,
                        kind=f"lab_{analysis}",
                    )
        except Exception as e:  # Broad: a failed file is reported, the others go on
            logger.exception(f"{path}: {type(e).__name__}: {e}")
            done.append(f"FAILED {path.name}: {type(e).__name__}: {e}")
    return done
