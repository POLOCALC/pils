"""Versioned lab analyses: common results handling.

Results follow the flight analyses' layout, with the test folder in place
of the flight folder and the measurement stem as the parent group::

    <item>/<test_type>/proc/<analysis>/
    ├── <analysis>_solution.h5
    │   └── <stem>/                    # e.g. 20251021_140000
    │       └── rev_YYYYMMDD_HHMMSS/   # one group per run (tables + metadata)
    └── <stem>/rev_YYYYMMDD_HHMMSS/    # plots of that run

Nothing is written until a run is saved, and ``output_dir`` can redirect
the results away from the data share.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import polars as pl

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
        """Results file under a test folder (or output dir)."""
        return (
            Path(results_root)
            / "proc"
            / cls.analysis_name
            / f"{cls.analysis_name}_solution.h5"
        )

    @property
    def results_path(self) -> Path:
        return self.results_dir / f"{self.analysis_name}_solution.h5"

    def figures_dir(self, version_name: str) -> Path:
        """Folder for the plots of a run (created when plots are saved)."""
        return self.results_dir / self.measurement.stem / version_name

    def list_versions(self) -> list[str]:
        """Runs of this measurement, oldest first."""
        return hdf5.list_versions(self.results_path, parent=self.measurement.stem)

    def get_version(self, version_name: str) -> LabVersion:
        """Load one run (KeyError if missing)."""
        tables, metadata = hdf5.load_version(
            self.results_path, version_name, parent=self.measurement.stem
        )
        return LabVersion(version_name, tables, metadata)

    def get_latest_version(self) -> LabVersion | None:
        """The most recent run, or None."""
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
        directory = results_dir or self.results_dir
        path = directory / f"{self.analysis_name}_solution.h5"
        hdf5.save_version(
            path,
            version.version_name,
            version.tables,
            version.metadata,
            parent=self.measurement.stem,
        )
        logger.info(f"Saved {self.analysis_name} {version.version_name} to {path}")
        return path
