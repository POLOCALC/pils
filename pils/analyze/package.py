"""Package a flight: raw and synchronized data plus the analyses, in ``proc/``.

One call takes a flight from its raw files to the products described in
``docs/user-guide/directory-structure.md``::

    flight_YYYYMMDD_hhmm/proc/
    ├── sync_data.h5                    # metadata, raw_data, sync_data/rev_*
    ├── azel/
    │   ├── azel_solution.h5            # <TELESCOPE>/rev_*/azel_data
    │   └── rev_YYYYMMDD_hhmmss/
    │       └── rev_YYYYMMDD_hhmmss_azel_<telescope>_<YYYYMMDD_hhmm>.csv
    ├── power/                          # only for the requested wafer slots
    └── package_manifest.json           # what the last packaging produced

Every run adds new revisions; earlier ones are kept. Steps that do not apply
(a sensor that is not on the payload, a telescope missing from the campaign
coordinates) are skipped; a step that fails is recorded and the remaining
steps still run, except that nothing runs without synchronized data.

Examples
--------
>>> from pils.analyze.package import FlightPackager
>>> packager = FlightPackager.from_flight("flight_20251208_1927")  # doctest: +SKIP
>>> result = packager.run(telescopes=["SATP1", "SATP3"])  # doctest: +SKIP
>>> print(result.summary())  # doctest: +SKIP
"""

import datetime
import json
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from pils.flight import Flight
from pils.utils.hdf5 import dumps_metadata, new_version_name
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)

# Payload sensors tried in order; the GPS is the synchronization reference
PAYLOAD_SENSORS = ("gps", "adc", "inclinometer", "lm76", "imu")
# Telescopes surveyed in the campaign coordinates file
DEFAULT_TELESCOPES = ("SATP1", "SATP2", "SATP3")
# Same rates as the power analysis, so it reuses this synchronization
SYNC_RATES = {"drone": 100.0, "litchi": 100.0, "inclinometer": 100.0, "gps": 100.0}
# Columns of the AZEL file handed to the telescopes
AZEL_CSV_COLUMNS = ("timestamp", "az", "el", "srange")

SYNC_FILE = "sync_data.h5"
MANIFEST_FILE = "package_manifest.json"


@dataclass
class StepResult:
    """Outcome of one packaging step.

    Attributes
    ----------
    name : str
        Step name, e.g. ``"sync"`` or ``"azel SATP1"``.
    status : str
        ``"done"``, ``"skipped"`` or ``"failed"``.
    detail : str
        What was done, or why it was skipped or failed.
    outputs : list[str]
        Files written by the step.
    """

    name: str
    status: str
    detail: str = ""
    outputs: list[str] = field(default_factory=list)


@dataclass
class PackageResult:
    """Outcome of packaging one flight."""

    flight_name: str
    version_name: str
    proc_dir: str
    saved: bool
    steps: list[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no step failed."""
        return all(step.status != "failed" for step in self.steps)

    def add(self, name: str, status: str, detail: str = "", outputs=()) -> None:
        self.steps.append(StepResult(name, status, detail, [str(p) for p in outputs]))
        log = logger.warning if status == "failed" else logger.info
        log(f"{self.flight_name} | {name}: {status} {detail}".rstrip())

    def summary(self) -> str:
        """Plain-text table of the steps."""
        width = max((len(s.name) for s in self.steps), default=4)
        lines = [f"{self.flight_name} ({self.version_name})"]
        for step in self.steps:
            lines.append(f"  {step.name:<{width}}  {step.status:<7}  {step.detail}")
            lines.extend(f"  {'':<{width}}           -> {p}" for p in step.outputs)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "ok": self.ok}


def flight_info_from_path(flight_dir: str | Path) -> dict[str, Any]:
    """Flight info for a flight folder that is not (yet) in STOUT.

    Parameters
    ----------
    flight_dir : str | Path
        ``.../flight_YYYYMMDD_hhmm`` with ``drone/`` and ``aux/`` inside.

    Raises
    ------
    FileNotFoundError
        If the folder has no ``drone`` folder.
    """
    flight_dir = Path(flight_dir).resolve()
    if not (flight_dir / "drone").is_dir():
        raise FileNotFoundError(f"No drone folder in {flight_dir}")
    info: dict[str, Any] = {
        "flight_id": None,
        "flight_name": flight_dir.name,
        "drone_data_folder_path": str(flight_dir / "drone"),
        "aux_data_folder_path": str(flight_dir / "aux"),
        "processed_data_folder_path": str(flight_dir / "proc"),
        "flight_info": {},
    }
    try:
        takeoff = datetime.datetime.strptime(flight_dir.name[7:], "%Y%m%d_%H%M")
        info["takeoff_datetime"] = takeoff.replace(tzinfo=datetime.UTC)
    except ValueError:
        pass
    return info


def resolve_flight(flight: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Flight info from a STOUT flight dict, a flight folder or a flight name.

    Raises
    ------
    LookupError
        If a name is not a folder and is not in STOUT.
    """
    if isinstance(flight, dict):
        return flight
    if Path(flight).is_dir():
        return flight_info_from_path(flight)
    from pils.loader.stout import StoutLoader

    info = StoutLoader().load_single_flight(flight_name=str(flight))
    if info is None:
        raise LookupError(f"Flight {flight} is not a folder and is not in STOUT")
    return info


class FlightPackager:
    """Build the ``proc/`` products of a flight from its raw files.

    Parameters
    ----------
    flight_info : dict
        STOUT flight dict or :func:`flight_info_from_path` output.
    output_dir : str | Path, optional
        Write to ``<output_dir>/<flight_name>/proc/`` instead of the flight
        folder (one subfolder per flight, so a batch does not collide).
    """

    def __init__(
        self, flight_info: dict[str, Any], output_dir: str | Path | None = None
    ) -> None:
        # Sensor loaders add their metadata under flight_info["flight_info"]
        self.flight_info = {**flight_info}
        self.flight_info["flight_info"] = dict(flight_info.get("flight_info") or {})
        self.flight_name = str(
            flight_info.get("flight_name")
            or Path(flight_info["drone_data_folder_path"]).parent.name
        )
        self.flight_path = Path(flight_info["drone_data_folder_path"]).parent
        # Base folder that replaces the flight folder for the analyses
        self.output_dir = (
            Path(output_dir) / self.flight_name if output_dir is not None else None
        )
        self.proc_dir = (self.output_dir or self.flight_path) / "proc"
        self.flight: Flight | None = None

    @classmethod
    def from_flight(
        cls, flight: str | Path | dict[str, Any], output_dir: str | Path | None = None
    ) -> "FlightPackager":
        """Packager for a flight name (STOUT), a flight folder or a flight dict."""
        return cls(resolve_flight(flight), output_dir=output_dir)

    # -- steps --------------------------------------------------------------

    def _load(self, result: PackageResult, sensors: Sequence[str]) -> Flight | None:
        flight = Flight(self.flight_info)
        sensors_dir = Path(self.flight_info["aux_data_folder_path"]) / "sensors"
        for sensor in sensors:
            try:
                flight.add_sensor_data(sensor)
            except Exception as e:  # a sensor missing from this payload
                result.add(f"load {sensor}", "skipped", _error(e))
                continue
            data = getattr(flight.raw_data.payload_data, sensor, None)
            if data is None or (isinstance(data, pl.DataFrame) and data.is_empty()):
                result.add(f"load {sensor}", "skipped", f"no data in {sensors_dir}")
            else:
                result.add(f"load {sensor}", "done", _describe(data))
        if getattr(flight.raw_data.payload_data, "gps", None) is None:
            result.add("load", "failed", "no payload GPS: it is the sync reference")
            return None
        try:
            flight.add_drone_data()
            result.add(
                "load drone", "done", _describe(flight.raw_data.drone_data.drone)
            )
            records = getattr(flight.raw_data.drone_data, "records", None) or {}
            if records:
                result.add("load drone records", "done", _describe(records))
        except Exception as e:
            result.add("load drone", "failed", _error(e))
        return flight

    def _sync(self, flight: Flight, result: PackageResult) -> bool:
        try:
            synced = flight.sync(
                use_rtk_data=True, common_time=True, target_rate=dict(SYNC_RATES)
            )
        except Exception as e:
            result.add("sync", "failed", _error(e))
            return False
        result.add(
            "sync", "done", ", ".join(f"{k} {_describe(v)}" for k, v in synced.items())
        )
        return True

    def _save_flight(
        self, flight: Flight, result: PackageResult, save: bool, sensors: Sequence[str]
    ) -> str | None:
        if not save:
            result.add("save sync_data.h5", "skipped", "dry run")
            return None
        path = self.proc_dir / SYNC_FILE
        try:
            flight.to_hdf5(
                path,
                sync_metadata={
                    "package_version": result.version_name,
                    "sync_method": "gps correlation",
                    "common_time": True,
                    "use_rtk_data": True,
                    "target_rate": dumps_metadata(SYNC_RATES),
                    "sensors_requested": list(sensors),
                },
            )
        except Exception as e:
            result.add("save sync_data.h5", "failed", _error(e))
            return None
        revision = _latest_sync_revision(path)
        result.add("save sync_data.h5", "done", f"sync_data/{revision}", [path])
        return revision

    def _azel(
        self,
        flight: Flight,
        result: PackageResult,
        telescopes: Iterable[str],
        dji_broadcast_geod: dict[str, float],
        save: bool,
        sync_revision: str | None,
    ) -> None:
        if save:
            self._run_azel(
                flight, result, telescopes, dji_broadcast_geod, save, sync_revision
            )
            return
        # AZELAnalysis creates its folder up front: keep a dry run out of proc/
        with tempfile.TemporaryDirectory() as scratch:
            self._run_azel(
                flight,
                result,
                telescopes,
                dji_broadcast_geod,
                save,
                sync_revision,
                output_dir=Path(scratch),
            )

    def _run_azel(
        self,
        flight: Flight,
        result: PackageResult,
        telescopes: Iterable[str],
        dji_broadcast_geod: dict[str, float],
        save: bool,
        sync_revision: str | None,
        output_dir: Path | None = None,
    ) -> None:
        from pils.analyze.azel import AZELAnalysis

        try:
            azel = AZELAnalysis(flight, output_dir=output_dir or self.output_dir)
        except Exception as e:
            result.add("azel", "failed", _error(e))
            return
        for telescope in telescopes:
            name = f"azel {telescope}"
            try:
                version = azel.run_analysis(
                    telescope,
                    dji_broadcast_geod,
                    save_data=save,
                    additional_metadata={
                        "package_version": result.version_name,
                        "sync_revision": sync_revision,
                    },
                )
            except ValueError as e:
                if "No telescope positions" in str(e):
                    result.add(name, "skipped", "not in the campaign coordinates")
                else:
                    result.add(name, "failed", f"ValueError: {e}")
                continue
            except Exception as e:
                result.add(name, "failed", _error(e))
                continue
            if version is None:
                result.add(name, "failed", "no valid RTK data after filtering")
                continue
            detail = f"{version.version_name}, {version.azel_data.height} rows"
            if not save:
                result.add(name, "done", f"{detail} (not saved)")
                continue
            csv_path = self._write_azel_csv(
                azel.azel_dir, version.version_name, telescope, version.azel_data
            )
            result.add(
                name, "done", detail, [azel.azel_dir / "azel_solution.h5", csv_path]
            )

    def _write_azel_csv(
        self, azel_dir: Path, version_name: str, telescope: str, data: pl.DataFrame
    ) -> Path:
        """The AZEL file for the telescope, in the version's folder."""
        folder = azel_dir / version_name
        folder.mkdir(parents=True, exist_ok=True)
        stem = self.flight_name.removeprefix("flight_")
        path = folder / f"{version_name}_azel_{telescope.lower()}_{stem}.csv"
        columns = [c for c in AZEL_CSV_COLUMNS if c in data.columns]
        data.select(columns).write_csv(path)
        return path

    def _power(
        self,
        flight: Flight,
        result: PackageResult,
        targets: Iterable[tuple[str, str]],
        save: bool,
        lab_root: str | Path | None,
    ) -> None:
        targets = list(targets)
        if not targets:
            return
        from pils.analyze.power import PowerAnalysis
        from pils.lab_analysis import LabStore

        for telescope, wafer_slot in targets:
            name = f"power {telescope}/{wafer_slot}"
            try:
                version = PowerAnalysis(
                    flight, output_dir=self.output_dir
                ).run_analysis(
                    telescope,
                    wafer_slot,
                    store=LabStore(lab_root),
                    save_data=save,
                    plot=save,
                )
            except Exception as e:
                result.add(name, "failed", _error(e))
                continue
            outputs = [self.proc_dir / "power" / "power_solution.h5"] if save else []
            result.add(name, "done", version.version_name, outputs)

    def _write_manifest(self, result: PackageResult, inputs: dict[str, Any]) -> None:
        import pils

        path = self.proc_dir / MANIFEST_FILE
        manifest = {
            "pils_version": getattr(pils, "__version__", "unknown"),
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "flight_id": self.flight_info.get("flight_id"),
            "inputs": inputs,
            **result.to_dict(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, default=str))
        result.steps.append(StepResult("manifest", "done", "", [str(path)]))

    # -- entry point --------------------------------------------------------

    def run(
        self,
        telescopes: Iterable[str] = DEFAULT_TELESCOPES,
        dji_broadcast_geod: dict[str, float] | None = None,
        power_targets: Iterable[tuple[str, str]] = (),
        sensors: Sequence[str] = PAYLOAD_SENSORS,
        lab_root: str | Path | None = None,
        save: bool = True,
    ) -> PackageResult:
        """Load, synchronize, save and analyse the flight.

        Parameters
        ----------
        telescopes : iterable of str
            Telescopes to compute AZEL for (names in the campaign coordinates).
        dji_broadcast_geod : dict, optional
            DJI RTK base position broadcast to the drone (``lat``, ``lon``,
            ``alt``); defaults to the telescopes config.
        power_targets : iterable of (telescope, wafer slot)
            Expected-power analyses to run, e.g. ``[("satp1", "ws0")]``.
        sensors : sequence of str
            Payload sensors to load; missing ones are skipped.
        lab_root : str | Path, optional
            Lab data root for the power analysis.
        save : bool
            Write the products; False computes everything without writing.

        Returns
        -------
        PackageResult
            Status of each step and the files written.
        """
        from pils.config.telescopes import TELESCOPES_CONFIG

        telescopes = list(telescopes)
        power_targets = list(power_targets)
        geod = dji_broadcast_geod or TELESCOPES_CONFIG["dji_broadcast_geod"]
        result = PackageResult(
            self.flight_name, new_version_name(), str(self.proc_dir), saved=save
        )

        flight = self._load(result, sensors)
        self.flight = flight
        if flight is not None and self._sync(flight, result):
            sync_revision = self._save_flight(flight, result, save, sensors)
            self._azel(flight, result, telescopes, geod, save, sync_revision)
            self._power(flight, result, power_targets, save, lab_root)

        if save:
            self._write_manifest(
                result,
                {
                    "telescopes": telescopes,
                    "dji_broadcast_geod": geod,
                    "power_targets": power_targets,
                    "sensors": list(sensors),
                    "sync_rates": SYNC_RATES,
                },
            )
        return result


def package_flight(
    flight: str | Path | dict[str, Any],
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> PackageResult:
    """Package one flight; see :meth:`FlightPackager.run` for the options."""
    return FlightPackager.from_flight(flight, output_dir=output_dir).run(**kwargs)


def _describe(data: Any) -> str:
    if isinstance(data, pl.DataFrame):
        return f"{data.height}x{data.width}"
    if isinstance(data, dict):
        return "{" + ", ".join(f"{k}: {_describe(v)}" for k, v in data.items()) + "}"
    return type(data).__name__


def _latest_sync_revision(path: Path) -> str | None:
    import h5py

    with h5py.File(path, "r") as f:
        if "sync_data" not in f:
            return None
        names = sorted(k for k in f["sync_data"] if k.startswith("rev_"))
    return names[-1] if names else None


def _error(e: Exception) -> str:
    """One-line description of an exception."""
    lines = str(e).strip().splitlines()
    message = lines[0] if lines else ""
    if len(message) > 160:
        message = message[:157] + "..."
    return f"{type(e).__name__}: {message}"
