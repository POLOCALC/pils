"""Versioned lab analyses: common results handling.

Results follow the flight analyses' layout (:mod:`pils.store`), with the
test folder in place of the flight folder and the measurement stem as the
parent folder::

    <item>/<test_type>/proc/<analysis>/
    └── <stem>/                        # e.g. 20251021_140000
        └── rev_YYYYMMDD_HHMMSS/       # one run: manifest.json,
                                       # <table>.parquet, plots

Runs saved before in ``<analysis>_solution.h5`` (one HDF5 group per stem and
run) are still listed and loaded; ``pils.migrate.migrate_lab`` converts them.

Nothing is written until a run is saved, and ``output_dir`` can redirect
the results away from the data share.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

from pils import store as result_store
from pils.lab_analysis.notes import LabMeasurement
from pils.utils import hdf5
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LabVersion:
    """One run of a lab analysis.

    Parameters
    ----------
    version_name : str
        ``rev_YYYYMMDD_HHMMSS``.
    tables : dict[str, pl.DataFrame]
        Result tables by name.
    metadata : dict[str, Any]
        Run configuration, inputs and summary values.
    """

    version_name: str
    tables: dict[str, pl.DataFrame]
    metadata: dict[str, Any]

    def __getitem__(self, table: str) -> pl.DataFrame:
        """Return a table of the run by name."""
        return self.tables[table]


class LabAnalysis:
    """Base class: one analysis of one lab measurement.

    Subclasses set ``analysis_name`` and implement ``run_analysis``.
    """

    analysis_name: ClassVar[str] = ""

    def __init__(
        self, measurement: LabMeasurement, output_dir: str | Path | None = None
    ) -> None:
        self.measurement = measurement
        base = Path(output_dir) if output_dir is not None else measurement.folder
        self.results_dir = base / "proc" / self.analysis_name

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        stem: str | None = None,
        output_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> "LabAnalysis":
        """Analysis of the measurement ``stem`` (latest if None) in ``folder``."""
        return cls(LabMeasurement.find(folder, stem), output_dir=output_dir, **kwargs)

    @classmethod
    def solution_path(cls, results_root: str | Path) -> Path:
        """Legacy HDF5 results file under a test folder (or output dir)."""
        return (
            Path(results_root)
            / "proc"
            / cls.analysis_name
            / f"{cls.analysis_name}_solution.h5"
        )

    @property
    def results_path(self) -> Path:
        """Legacy HDF5 results file (runs saved before :mod:`pils.store`)."""
        return self.results_dir / f"{self.analysis_name}_solution.h5"

    def figures_dir(self, version_name: str) -> Path:
        """Folder for the plots of a run (created when plots are saved)."""
        return self.results_dir / self.measurement.stem / version_name

    def list_versions(self) -> list[str]:
        """List the runs of this measurement, oldest first (saved and legacy HDF5)."""
        stem = self.measurement.stem
        return sorted(
            set(result_store.list_versions(self.results_dir, parent=stem))
            | set(hdf5.list_versions(self.results_path, parent=stem))
        )

    def get_version(self, version_name: str) -> LabVersion:
        """Load one run (KeyError if missing)."""
        tables, metadata = load_run(
            self.results_dir, self.analysis_name, self.measurement.stem, version_name
        )
        return LabVersion(version_name, tables, metadata)

    def get_latest_version(self) -> LabVersion | None:
        """Return the most recent run, or None."""
        versions = self.list_versions()
        return self.get_version(versions[-1]) if versions else None

    def _base_metadata(self) -> dict[str, Any]:
        m = self.measurement
        return {
            "analysis": self.analysis_name,
            "test_type": m.test_type,
            "measurement": m.stem,
            "raw_file": str(m.raw_path) if m.raw_path else None,
            "notes_file": str(m.notes_path) if m.notes_path else None,
            "notes_config": m.config,
        }

    def _save(self, version: LabVersion, results_dir: Path | None = None) -> Path:
        """Save a run: ``<results_dir>/<stem>/<version>/`` (see :mod:`pils.store`)."""
        path = result_store.save_version(
            results_dir or self.results_dir,
            version.version_name,
            dict(version.tables),
            version.metadata,
            parent=self.measurement.stem,
            kind=f"lab_{self.analysis_name}",
        )
        logger.info(f"Saved {self.analysis_name} {version.version_name} to {path}")
        return path


def load_run(
    results_dir: Path, analysis: str, stem: str, version_name: str
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    """Tables and metadata of a run: saved (pils.store) or legacy HDF5.

    Raises
    ------
    KeyError
        If the run is in neither.
    """
    if version_name in result_store.list_versions(results_dir, parent=stem):
        tables, metadata = result_store.load_version(
            results_dir, version_name, parent=stem
        )
        return tables, metadata  # type: ignore[return-value]
    legacy = results_dir / f"{analysis}_solution.h5"
    if not legacy.is_file():
        raise KeyError(f"Version '{version_name}' not found in {results_dir / stem}")
    return hdf5.load_version(legacy, version_name, parent=stem)


def latest_run(
    results_dir: Path, analysis: str
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]] | None:
    """Latest run of the latest measurement with runs, saved or legacy HDF5."""
    legacy = results_dir / f"{analysis}_solution.h5"
    stems: set[str] = set()
    if results_dir.is_dir():
        stems |= {p.name for p in results_dir.iterdir() if p.is_dir()}
    if legacy.is_file():
        import h5py

        with h5py.File(legacy, "r") as f:
            stems |= set(f.keys())
    for stem in sorted(stems, reverse=True):  # latest measurement first
        versions = sorted(
            set(result_store.list_versions(results_dir, parent=stem))
            | set(hdf5.list_versions(legacy, parent=stem))
        )
        if versions:
            return load_run(results_dir, analysis, stem, versions[-1])
    return None
