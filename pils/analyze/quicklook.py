"""Quick look at a flight: one Markdown report of its data and their quality.

:class:`QuickLook` loads a flight's raw files and writes, in a new revision::

    flight_YYYYMMDD_hhmm/proc/quicklook/rev_YYYYMMDD_hhmmss/
    ├── report.md            # the report (links the plots below)
    ├── plots/*.png
    ├── <table>.parquet      # the report's tables (inventory, sensors, ...)
    └── manifest.json        # written last, as in pils.store

The report has a warnings summary first, then one section per topic: the
flight (STOUT), the files, the drone, the payload sensors, the
synchronisation, the GNSS data for PPK (rover and base) and the AZEL of the
telescopes. A section whose data is missing says why instead.

Nothing is changed in the flight folder except ``proc/quicklook``; with
``output_dir`` the report goes to ``<output_dir>/<flight>/proc/quicklook``.
The PPK tuning (``ppk_template``) runs in a temporary folder and only its
results are kept in the report's revision.

Examples
--------
>>> from pils.analyze.quicklook import QuickLook
>>> report = QuickLook.from_flight("flight_20251208_1927").run()  # doctest: +SKIP
>>> print(report)  # doctest: +SKIP
.../flight_20251208_1927/proc/quicklook/rev_20261017_180102/report.md
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils import store
from pils.analyze.package import DEFAULT_TELESCOPES, PAYLOAD_SENSORS, resolve_flight
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)

QUICKLOOK_DIR = "quicklook"
REPORT_FILE = "report.md"

# The drone is airborne above this height over the take-off point (m)
AIRBORNE_HEIGHT_M = 1.0
# A sensor gap longer than this is reported (s)
MAX_GAP_S = 1.0
# A sensor covering less of the airborne time than this is flagged (%)
MIN_COVERAGE_PCT = 95.0
# A time offset correlation below this is flagged
MIN_CORRELATION = 0.9
# A clock offset larger than this is flagged (s)
MAX_CLOCK_OFFSET_S = 60.0
# A Litchi log overlapping the drone's airborne time less than this is
# flagged as the log of another flight (%)
MIN_LITCHI_OVERLAP_PCT = 80.0
# SNR levels (dB-Hz) at which the satellites are counted
SNR_LEVELS = (30, 35, 40)
# DJI RTK position flag (rtk_pos_flag_1): solution type
DJI_RTK_FLAGS = {50: "fixed", 34: "float", 16: "single", 0: "none"}
# Series colours, in this order (light theme)
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6")


# -- small helpers ----------------------------------------------------------


def _seconds(frame: pl.DataFrame) -> np.ndarray | None:
    """UNIX seconds of a table's ``timestamp`` (float or datetime), or None."""
    if "timestamp" not in frame.columns or frame.is_empty():
        return None
    column = frame["timestamp"]
    if column.dtype.is_temporal():
        values = column.dt.epoch("us").cast(pl.Float64) / 1e6
    elif column.dtype.is_numeric():
        values = column.cast(pl.Float64)
    else:
        return None
    array = values.drop_nulls().to_numpy()
    return array[np.isfinite(array)] if array.size else None


def _utc(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "-"
    moment = datetime.datetime.fromtimestamp(seconds, datetime.timezone.utc)
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return "-" if not np.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def _size(megabytes: float | None) -> str:
    if megabytes is None:
        return "-"
    if megabytes >= 1000:
        return f"{megabytes / 1000:.1f} GB"
    if megabytes >= 1:
        return f"{megabytes:.1f} MB"
    return f"{megabytes * 1000:.0f} kB"


def md_table(rows: list[dict[str, Any]], digits: int = 1) -> str:
    """Markdown table of rows (dicts with the same keys)."""
    if not rows:
        return "_none_"
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join(_fmt(row.get(c), digits) for c in columns) + " |"
        )
    return "\n".join(lines)


def local_en(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """East and north (m) from the first point, small-area approximation."""
    radius = 6_371_000.0
    lat0, lon0 = np.radians(lat[0]), np.radians(lon[0])
    east = radius * np.cos(lat0) * (np.radians(lon) - lon0)
    north = radius * (np.radians(lat) - lat0)
    return east, north


# -- data checks (pure functions, tested on their own) ----------------------


def airborne_interval(
    times: np.ndarray, height: np.ndarray, threshold: float = AIRBORNE_HEIGHT_M
) -> tuple[float, float] | None:
    """Longest stretch with the height above ``threshold`` (UNIX s)."""
    above = np.asarray(height) > threshold
    if not above.any():
        return None
    edges = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    k = int(np.argmax(times[stops] - times[starts]))
    return float(times[starts[k]]), float(times[stops[k]])


def table_summary(
    times: np.ndarray, interval: tuple[float, float] | None
) -> dict[str, Any]:
    """Samples, span, rate, gaps and coverage of the airborne interval."""
    times = np.sort(times)
    steps = np.diff(times)
    # A gap: longer than 1 s and than 5 sampling steps (slow sensors)
    limit = max(MAX_GAP_S, 5 * float(np.median(steps))) if steps.size else MAX_GAP_S
    gaps = steps[steps > limit]
    summary: dict[str, Any] = {
        "samples": int(times.size),
        "start_utc": _utc(float(times[0])),
        "duration_s": float(times[-1] - times[0]),
        "rate_hz": float(1 / np.median(steps)) if steps.size else float("nan"),
        "gaps": int(gaps.size),
        "longest_gap_s": float(gaps.max()) if gaps.size else 0.0,
        "coverage_pct": float("nan"),
    }
    if interval is not None and interval[1] > interval[0]:
        start, end = interval
        inside = times[(times >= start) & (times <= end)]
        if inside.size:
            # Covered time: the span minus the gaps, inside the interval
            edges = np.concatenate([[start], inside, [end]])
            missing = np.diff(edges)
            covered = (end - start) - missing[missing > limit].sum()
            summary["coverage_pct"] = float(100 * max(covered, 0) / (end - start))
        else:
            summary["coverage_pct"] = 0.0
    return summary


def overlap_pct(span: tuple[float, float], interval: tuple[float, float]) -> float:
    """Share of ``interval`` (%) inside ``span``."""
    overlap = min(span[1], interval[1]) - max(span[0], interval[0])
    return float(100 * max(overlap, 0.0) / max(interval[1] - interval[0], 1e-9))


def rtk_flag_summary(flags: pl.Series) -> dict[str, float]:
    """Share (%) of each DJI RTK solution type."""
    values = flags.drop_nulls().to_numpy()
    if not values.size:
        return {}
    out = {
        name: float(100 * np.mean(values == code))
        for code, name in DJI_RTK_FLAGS.items()
    }
    out["other"] = float(100 - sum(out.values()))
    return out


def offset_rows(offsets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per synchronisation offset."""
    rows = []
    for source, result in offsets.items():
        rows.append(
            {
                "source": source,
                "offset_s": float(result.get("time_offset", float("nan"))),
                "correlation": result.get("correlation"),
                "method": result.get("method", "correlation"),
                "spatial_offset_m": result.get("spatial_offset_m"),
            }
        )
    return rows


def offset_warnings(offsets: dict[str, dict[str, Any]]) -> list[str]:
    """What looks wrong in the synchronisation offsets."""
    warnings = []
    for source, result in offsets.items():
        corr = result.get("correlation")
        if corr is not None and corr < MIN_CORRELATION:
            warnings.append(
                f"Sync: {source} correlation {corr:.2f} (< {MIN_CORRELATION}): "
                "check the offset"
            )
        offset = result.get("time_offset")
        if (
            offset is not None
            and source in ("drone_gps", "litchi_gps")
            and abs(offset) > MAX_CLOCK_OFFSET_S
        ):
            warnings.append(f"Sync: {source} clock off by {offset:.0f} s")
    if offsets.get("payload_clock", {}).get("method") == "none":
        warnings.append(
            "Sync: no offset for the payload clock (ADC, LM76, IMU taken as GPS time)"
        )
    return warnings


# -- RINEX -------------------------------------------------------------------


def read_rinex(path: str | Path) -> tuple[list[str], list[dict[str, dict]]]:
    """Epoch times and, per epoch, ``{"G05": {"1": (snr, lli), ...}}``.

    SNR is None when missing; LLI is the loss-of-lock bit of the band's phase.
    """
    types: dict[str, list[str]] = {}
    system = ""
    with open(path, errors="replace") as fh:
        for line in fh:
            label = line[60:].strip()
            if label == "SYS / # / OBS TYPES":
                if line[0] != " ":
                    system = line[0]
                    types[system] = []
                types[system] += line[7:58].split()
            if label == "END OF HEADER":
                break
        times: list[str] = []
        epochs: list[dict[str, dict]] = []
        current: dict[str, dict] = {}
        for line in fh:
            if line.startswith(">"):
                current = {}
                epochs.append(current)
                times.append(line[2:21])
                continue
            obs = types.get(line[0])
            if obs is None or not epochs:
                continue
            bands: dict[str, list] = {}
            for j, kind in enumerate(obs):
                if kind[0] not in "LS":
                    continue
                text = line[3 + 16 * j : 3 + 16 * j + 16]
                if not text[:14].strip():
                    continue
                entry = bands.setdefault(kind[1], [None, 0])
                if kind[0] == "S":
                    value = float(text[:14])
                    entry[0] = value if value > 0 else None
                else:
                    flag = text[14:15].strip()
                    entry[1] = int(flag) & 1 if flag.isdigit() else 0
            current[line[:3]] = {b: tuple(v) for b, v in bands.items()}
    return times, epochs


def band_table(
    epochs: list[dict[str, dict]], interval_s: float = 1.0
) -> list[dict[str, Any]]:
    """Per system and band: satellites (all, and above each SNR level), SNR,
    loss of lock and continuous tracking (observation-weighted arc, s)."""
    keys = sorted(
        {
            (sat[0], band)
            for epoch in epochs
            for sat, obs in epoch.items()
            for band in obs
        }
    )
    rows = []
    for system, band in keys:
        counts: dict[int, list[int]] = {level: [] for level in (0, *SNR_LEVELS)}
        snrs: list[float] = []
        flagged = observed = 0
        running: dict[str, int] = {}
        arcs: list[int] = []
        for epoch in epochs:
            values = []
            seen = set()
            for sat, obs in epoch.items():
                if sat[0] != system or band not in obs or obs[band][0] is None:
                    continue
                snr, lli = obs[band]
                values.append(snr)
                observed += 1
                flagged += lli
                seen.add(sat)
            for level in counts:
                counts[level].append(sum(v >= level for v in values))
            snrs += values
            for sat in list(running):
                if sat not in seen:
                    arcs.append(running.pop(sat))
            for sat in seen:
                running[sat] = running.get(sat, 0) + 1
        arcs += running.values()
        if not snrs:
            continue
        row: dict[str, Any] = {"system": system, "band": band}
        row["sats"] = float(np.median(counts[0]))
        for level in SNR_LEVELS:
            row[f"sats_{level}dbhz"] = float(np.median(counts[level]))
        row["snr_dbhz"] = float(np.median(snrs))
        row["snr_p10_dbhz"] = float(np.percentile(snrs, 10))
        row["lli_pct"] = 100 * flagged / observed if observed else float("nan")
        row["arc_s"] = (
            float(np.sum(np.square(arcs)) / np.sum(arcs)) * interval_s if arcs else 0.0
        )
        rows.append(row)
    return rows


def common_satellites(
    rover: tuple[list[str], list[dict[str, dict]]],
    base: tuple[list[str], list[dict[str, dict]]],
) -> dict[str, float]:
    """Median satellites seen by both receivers at the same epoch, by system."""
    base_by_time = {t[:19]: epoch for t, epoch in zip(*base, strict=True)}
    counts: dict[str, list[int]] = {}
    for t, epoch in zip(*rover, strict=True):
        other = base_by_time.get(t[:19])
        if other is None:
            continue
        both = set(epoch) & set(other)
        for system in {s[0] for s in epoch} | {s[0] for s in other}:
            counts.setdefault(system, []).append(sum(s[0] == system for s in both))
    return {s: float(np.median(v)) for s, v in sorted(counts.items())}


def _rinex_time(text: str) -> np.datetime64:
    parts = text.split()
    year, month, day, hour, minute = (int(p) for p in parts[:5])
    second = int(float(parts[5]))
    return np.datetime64(
        f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"
    )


# -- the report ----------------------------------------------------------------


@dataclass
class Section:
    """A report section: a title and Markdown lines."""

    title: str
    lines: list[str] = field(default_factory=list)

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)


class _WarningCollector(logging.Handler):
    """Collects the pils warnings logged while the data load."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage().strip()
        if message and message not in self.messages:
            self.messages.append(message)


@contextlib.contextmanager
def _collect_warnings() -> Iterator[_WarningCollector]:
    handler = _WarningCollector()
    root = logging.getLogger("pils")
    root.addHandler(handler)
    try:
        yield handler
    finally:
        root.removeHandler(handler)


@contextlib.contextmanager
def _capture_offsets() -> Iterator[dict[str, dict[str, Any]]]:
    """The Synchronizer's full offsets (method, correlation) during a sync;
    ``Flight.sync_offsets`` keeps the seconds only."""
    from pils.synchronizer import Synchronizer

    captured: dict[str, dict[str, Any]] = {}
    original = Synchronizer.synchronize

    def synchronize(self: Synchronizer, *args: Any, **kwargs: Any) -> Any:
        try:
            return original(self, *args, **kwargs)
        finally:
            captured.update(self.offsets)

    Synchronizer.synchronize = synchronize  # type: ignore[method-assign]
    try:
        yield captured
    finally:
        Synchronizer.synchronize = original  # type: ignore[method-assign]


class QuickLook:
    """Quick-look report of one flight.

    Parameters
    ----------
    flight_info : dict
        STOUT flight dict or :func:`pils.analyze.package.flight_info_from_path`.
    output_dir : str | Path, optional
        Write to ``<output_dir>/<flight_name>/proc/quicklook/`` instead of the
        flight folder.
    """

    def __init__(
        self, flight_info: dict[str, Any], output_dir: str | Path | None = None
    ) -> None:
        self.flight_info = {**flight_info}
        self.flight_info["flight_info"] = dict(flight_info.get("flight_info") or {})
        self.flight_path = Path(flight_info["drone_data_folder_path"]).parent
        self.flight_name = str(flight_info.get("flight_name") or self.flight_path.name)
        base = (
            Path(output_dir) / self.flight_name
            if output_dir is not None
            else self.flight_path
        )
        self.quicklook_dir = base / "proc" / QUICKLOOK_DIR
        self.warnings: list[str] = []
        self.sections: list[Section] = []
        self.tables: store.Tables = {}
        self.metadata: dict[str, Any] = {}
        self.flight: Any = None
        self.airborne: tuple[float, float] | None = None
        self.revision_dir: Path | None = None

    @classmethod
    def from_flight(
        cls, flight: str | Path | dict[str, Any], output_dir: str | Path | None = None
    ) -> QuickLook:
        """Quick look for a flight name (STOUT), a flight folder or a dict."""
        return cls(resolve_flight(flight), output_dir=output_dir)

    # -- entry point ----------------------------------------------------------

    def run(
        self,
        telescopes: Iterable[str] = DEFAULT_TELESCOPES,
        rinex: bool = True,
        ppk_template: str | Path | None = None,
    ) -> Path:
        """Load the flight, check its data and write the report.

        Parameters
        ----------
        telescopes : iterable of str
            Telescopes for the AZEL summary.
        rinex : bool, default=True
            Convert the rover and base recordings (RTKLIB ``convbin``) and
            profile them for PPK.
        ppk_template : str | Path, optional
            RTKLIB configuration: also tune it to the data
            (:func:`pils.analyze.ppk_tuning.tune`, validated on the drone RTK
            track) and report the result. Takes several minutes.

        Returns
        -------
        Path
            The report (``report.md``).
        """
        self.revision_dir = self.quicklook_dir / store.new_version_name()
        self.plots_dir = self.revision_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        steps = [
            ("Flight", self._flight_section),
            ("Files", self._inventory_section),
            ("Drone", self._drone_section),
            ("Payload sensors", self._sensor_section),
            ("Synchronisation", self._sync_section),
            (
                "GNSS data for PPK",
                lambda s: self._gnss_section(s, rinex, ppk_template),
            ),
            ("AZEL", lambda s: self._azel_section(s, telescopes)),
        ]
        for title, step in steps:
            section = Section(title)
            try:
                step(section)
            except Exception as e:  # one section failing does not stop the rest
                logger.exception(f"Quick look {title} failed")
                section.add(f"Could not be done: `{type(e).__name__}: {e}`")
                self.warnings.append(f"{title}: failed ({type(e).__name__}: {e})")
            self.sections.append(section)

        report = self.revision_dir / REPORT_FILE
        report.write_text(self.markdown())
        store.write_tables(
            self.revision_dir,
            self.tables,
            {
                "flight_name": self.flight_name,
                "flight_id": self.flight_info.get("flight_id"),
                "warnings": self.warnings,
                "report": REPORT_FILE,
                "ppk_template": str(ppk_template) if ppk_template else None,
                **self.metadata,
            },
            kind="quicklook",
        )
        logger.info(f"Quick look report: {report}")
        return report

    def markdown(self) -> str:
        """The report as Markdown."""
        code = store.code_version()
        lines = [
            f"# Quick look: {self.flight_name}",
            "",
            f"Made {datetime.datetime.now():%Y-%m-%d %H:%M} with pils "
            f"{code['version']}"
            + (f" ({code['git_commit'][:7]})" if code.get("git_commit") else "")
            + ". Times are UTC.",
            "",
            "## Warnings",
            "",
        ]
        lines += [f"- {w}" for w in self.warnings] or ["None."]
        for section in self.sections:
            lines += ["", f"## {section.title}", "", *section.lines]
        return "\n".join(lines) + "\n"

    # -- sections -------------------------------------------------------------

    def _flight_section(self, section: Section) -> None:
        info = self.flight_info
        extra = info["flight_info"]
        rows = [
            ("Flight", self.flight_name),
            ("Campaign folder", self.flight_path.parent.parent.name),
            ("In STOUT", "yes" if info.get("flight_id") else "no"),
            ("Take-off (STOUT)", info.get("takeoff_datetime")),
            ("Landing (STOUT)", info.get("landing_datetime")),
            ("Type", info.get("flight_type")),
            ("Frequency (GHz)", extra.get("frequency")),
            ("Payload id", info.get("payload_id")),
            ("Flight plan", extra.get("flight_plan")),
            ("Notes", extra.get("notes")),
            ("Drone folder", info.get("drone_data_folder_path")),
            ("Aux folder", info.get("aux_data_folder_path")),
        ]
        section.add(
            md_table([{"": k, "value": v} for k, v in rows if v not in (None, "")])
        )
        if not info.get("flight_id"):
            section.add(
                "",
                "Not found in STOUT (run on a folder): take-off from the folder name.",
            )

    def _inventory_section(self, section: Section) -> None:
        drone = Path(self.flight_info["drone_data_folder_path"])
        aux = Path(self.flight_info["aux_data_folder_path"])
        day_base = self.flight_path.parent / "base"
        campaign_base = self.flight_path.parent.parent / "base"
        rows: list[dict[str, Any]] = []
        for label, folder in [
            ("drone", drone),
            ("aux", aux),
            ("base (day)", day_base),
            ("base (campaign)", campaign_base),
        ]:
            if not folder.is_dir():
                rows.append({"folder": label, "file": "(missing)", "size_mb": None})
                continue
            for path in sorted(p for p in folder.rglob("*") if p.is_file()):
                size = path.stat().st_size
                name = str(path.relative_to(folder))
                rows.append({"folder": label, "file": name, "size_mb": size / 1e6})
                if size == 0:
                    self.warnings.append(f"Files: {label}/{name} is empty")
        self.tables["inventory"] = pl.DataFrame(
            rows, schema={"folder": pl.String, "file": pl.String, "size_mb": pl.Float64}
        )
        # Long base folders: one line per folder, files counted
        shown = [r for r in rows if not r["folder"].startswith("base")]
        for label in ("base (day)", "base (campaign)"):
            files = [r for r in rows if r["folder"] == label]
            if files and files[0]["file"] == "(missing)":
                shown.append(files[0])
            elif files:
                total = sum(r["size_mb"] or 0 for r in files)
                shown.append(
                    {"folder": label, "file": f"{len(files)} files", "size_mb": total}
                )
        section.add(
            md_table(
                [
                    {
                        "folder": r["folder"],
                        "file": r["file"],
                        "size": _size(r["size_mb"]),
                    }
                    for r in shown
                ]
            )
        )
        if not any(drone.glob("*")):
            self.warnings.append("Files: the drone folder is empty")
        if not (aux / "sensors").is_dir():
            self.warnings.append("Files: no aux/sensors folder")

        # Load the data now; the next sections use it
        from pils.flight import Flight

        self.flight = Flight(self.flight_info)
        with _collect_warnings() as collected:
            loaded = []
            for sensor in PAYLOAD_SENSORS:
                try:
                    self.flight.add_sensor_data(sensor)
                    loaded.append(sensor)
                except Exception as e:
                    logger.info(f"{sensor}: {e}")
            try:
                self.flight.add_drone_data()
            except Exception as e:
                self.warnings.append(f"Drone: could not load ({type(e).__name__}: {e})")
        if collected.messages:
            section.add("", "Messages while loading:", "")
            section.add(*(f"- {m}" for m in collected.messages))
            self.warnings += [
                f"Load: {m}" for m in collected.messages if "stopped early" in m
            ]

    # The drone ---------------------------------------------------------------

    def _height_track(self) -> tuple[np.ndarray, np.ndarray, str] | None:
        """Time and height above take-off, from the DJI OSD or Litchi."""
        drone = self.flight.raw_data.drone_data
        records = getattr(drone, "records", None) or {}
        sources = [("DJI OSD", records.get("osd")), ("Litchi", drone.litchi)]
        for name, table in sources:
            if not isinstance(table, pl.DataFrame) or "h_rel_m" not in table.columns:
                continue
            table = table.filter(
                pl.col("timestamp").is_not_null() & pl.col("h_rel_m").is_not_null()
            ).sort("timestamp")
            times = _seconds(table)
            if times is not None and times.size == table.height:
                return times, table["h_rel_m"].to_numpy(), name
        return None

    def _drone_section(self, section: Section) -> None:
        drone = self.flight.raw_data.drone_data
        table = drone.drone if isinstance(drone.drone, pl.DataFrame) else None
        if table is None or table.is_empty():
            section.add("No drone data.")
            self.warnings.append("Drone: no data")
            return
        track = self._height_track()
        if track is None:
            section.add("No height above take-off (DJI OSD or Litchi).")
        else:
            times, height, source = track
            self.airborne = airborne_interval(times, height)
        rows: list[dict[str, Any]] = []
        if self.airborne is None:
            self.warnings.append("Drone: no airborne interval found")
        else:
            start, end = self.airborne
            inside = (times >= start) & (times <= end)
            rows += [
                {"": "Airborne", "value": f"{_utc(start)} - {_utc(end)[11:]}"},
                {"": "Airborne time (min)", "value": f"{(end - start) / 60:.1f}"},
                {
                    "": f"Max height ({source}, m)",
                    "value": f"{height[inside].max():.1f}",
                },
            ]
            self.metadata["airborne_utc"] = [_utc(start), _utc(end)]
            stout_takeoff = self.flight_info.get("takeoff_datetime")
            if isinstance(stout_takeoff, datetime.datetime):
                if stout_takeoff.tzinfo is None:
                    stout_takeoff = stout_takeoff.replace(tzinfo=datetime.timezone.utc)
                late = start - stout_takeoff.timestamp()
                if abs(late) > 10 * 60:
                    self.warnings.append(
                        f"Drone: take-off {late / 60:+.0f} min from the STOUT/folder "
                        "take-off time"
                    )

        lat_col = "rtk_lat_deg" if "rtk_lat_deg" in table.columns else "lat_deg"
        lon_col = lat_col.replace("lat", "lon")
        positions = table.filter(
            pl.col(lat_col).is_not_null() & (pl.col(lat_col) != 0)
        ).sort("timestamp")
        if self.airborne is not None:
            positions = positions.filter(pl.col("timestamp").is_between(*self.airborne))
        east = north = None
        if positions.height:
            east, north = local_en(
                positions[lat_col].to_numpy(), positions[lon_col].to_numpy()
            )
            rows += [
                {
                    "": "Positions from",
                    "value": lat_col.replace("_lat_deg", "") or "gps",
                },
                {"": "East extent (m)", "value": f"{np.ptp(east):.0f}"},
                {"": "North extent (m)", "value": f"{np.ptp(north):.0f}"},
                {
                    "": "Farthest from take-off (m)",
                    "value": f"{np.hypot(east, north).max():.0f}",
                },
            ]
        if "rtk_pos_flag_1" in positions.columns:
            flags = rtk_flag_summary(positions["rtk_pos_flag_1"])
            if flags:
                rows.append(
                    {
                        "": "RTK solution (airborne, %)",
                        "value": ", ".join(
                            f"{k} {v:.1f}" for k, v in flags.items() if v > 0
                        ),
                    }
                )
                self.metadata["drone_rtk_fixed_pct"] = flags["fixed"]
                if flags["fixed"] < 95:
                    self.warnings.append(
                        f"Drone: RTK fixed only {flags['fixed']:.0f} % of the "
                        "airborne time (the AZEL and PPK validation use it)"
                    )
        battery = (getattr(drone, "records", None) or {}).get("battery_info")
        if isinstance(battery, pl.DataFrame) and "charge_pct" in battery.columns:
            charged = battery.filter(pl.col("charge_pct") > 0)
            if charged.height:
                rows.append(
                    {
                        "": "Battery (%, V)",
                        "value": f"{charged['charge_pct'][0]} -> "
                        f"{charged['charge_pct'][-1]} %, "
                        f"{charged['pack_voltage_v'][0]:.1f} -> "
                        f"{charged['pack_voltage_v'][-1]:.1f} V",
                    }
                )
        section.add(md_table(rows))
        self._litchi_check(section)

        if east is not None and north is not None:
            minutes = (
                positions["timestamp"].to_numpy() - positions["timestamp"][0]
            ) / 60
            self._plot_track(east, north, minutes)
            self._link(section, "drone_track", "Track")
        if track is not None:
            self._plot_lines(
                "drone_height",
                [(f"{source} height", times, height)],
                "Height above take-off (m)",
            )
            self._link(section, "drone_height", "Height")

    def _litchi_check(self, section: Section) -> None:
        litchi = self.flight.raw_data.drone_data.litchi
        if not isinstance(litchi, pl.DataFrame) or litchi.is_empty():
            section.add("", "No Litchi log.")
            return
        flying = litchi
        if "is_flying" in litchi.columns and litchi["is_flying"].sum():
            flying = litchi.filter(pl.col("is_flying") > 0)
        times = _seconds(flying)
        if times is None:
            return
        span = (float(times.min()), float(times.max()))
        line = f"Litchi log: flying {_utc(span[0])} - {_utc(span[1])[11:]}"
        if self.airborne is not None:
            share = overlap_pct(span, self.airborne)
            line += f", {share:.0f} % of the drone's airborne time"
            self.metadata["litchi_overlap_pct"] = share
            if share < MIN_LITCHI_OVERLAP_PCT:
                self.warnings.append(
                    f"Files: the Litchi log covers {share:.0f} % of the drone's "
                    "airborne time: it may be the log of another flight"
                )
        section.add("", line + ".")

    # The payload ---------------------------------------------------------------

    def _sensor_tables(self) -> dict[str, pl.DataFrame]:
        """Every loaded payload table, by name."""
        payload = self.flight.raw_data.payload_data
        out: dict[str, pl.DataFrame] = {}
        for sensor in PAYLOAD_SENSORS:
            data = getattr(payload, sensor, None)
            data = getattr(data, "data", data)
            if isinstance(data, pl.DataFrame):
                out[sensor] = data
            elif isinstance(data, dict):
                for name, table in data.items():
                    if isinstance(table, pl.DataFrame):
                        out[f"{sensor}_{name}"] = table
        return out

    def _sensor_section(self, section: Section) -> None:
        tables = self._sensor_tables()
        if not tables:
            section.add("No payload sensor data.")
            self.warnings.append("Payload: no sensor data")
            return
        rows = []
        for name, table in tables.items():
            times = _seconds(table)
            if times is None:
                rows.append({"table": name, "samples": table.height})
                continue
            summary = table_summary(times, self.airborne)
            rows.append({"table": name, **summary})
            coverage = summary["coverage_pct"]
            if np.isfinite(coverage) and coverage < MIN_COVERAGE_PCT:
                self.warnings.append(
                    f"Payload: {name} covers {coverage:.0f} % of the airborne time"
                )
        missing = [
            s for s in ("gps", "adc") if not any(k.startswith(s) for k in tables)
        ]
        for sensor in missing:
            self.warnings.append(f"Payload: no {sensor} data")
        section.add(
            "Raw timestamps (before synchronisation); coverage is the share of "
            "the drone's airborne time with data. A gap is longer than 1 s and "
            "than 5 sampling steps.",
            "",
            md_table(rows),
        )
        self.tables["sensors"] = pl.DataFrame(rows)
        self._sensor_plots(section, tables)

    def _sensor_plots(self, section: Section, tables: dict[str, pl.DataFrame]) -> None:
        def series(name: str, column: str, scale: float = 1.0):
            table = tables.get(name)
            if table is None or column not in table.columns:
                return None
            table = table.filter(pl.col(column).is_not_null())
            times = _seconds(table)
            if times is None or times.size != table.height:
                return None
            return times, table[column].to_numpy() * scale

        plots = [
            ("adc", "ADC amplitude (mV)", [("amplitude", "adc", "amplitude_mv", 1)]),
            (
                "inclinometer",
                "Inclinometer angle (deg)",
                [
                    ("pitch", "inclinometer", "pitch_deg", 1),
                    ("roll", "inclinometer", "roll_deg", 1),
                    ("pitch", "inclinometer_ins", "pitch_rad", np.degrees(1)),
                    ("roll", "inclinometer_ins", "roll_rad", np.degrees(1)),
                ],
            ),
            ("lm76", "LM76 temperature (°C)", [("LM76", "lm76", "temperature_c", 1)]),
            (
                "imu",
                "Payload IMU",
                [
                    ("acc z (m/s²)", "imu_accelerometer", "acc_z_m_s2", 1),
                    ("gyro z (deg/s)", "imu_gyroscope", "gyro_z_rad_s", np.degrees(1)),
                    ("pressure (hPa)", "imu_barometer", "pressure_hpa", 1),
                ],
            ),
        ]
        for name, ylabel, wanted in plots:
            lines = []
            for label, table, column, scale in wanted:
                found = series(table, column, scale)
                if found is not None:
                    lines.append((label, *found))
            if name == "imu" and lines:
                lines = lines[:1]  # one quantity (one unit) per plot
                ylabel = f"Payload IMU {lines[0][0]}"
            if lines:
                self._plot_lines(f"payload_{name}", lines, ylabel)
                self._link(section, f"payload_{name}", ylabel)

    # Synchronisation -------------------------------------------------------------

    def _sync_section(self, section: Section) -> None:
        if "gps" not in self.flight.raw_data.payload_data:
            section.add("No payload GPS: it is the time reference, no sync.")
            self.warnings.append("Sync: not done (no payload GPS)")
            return
        with _capture_offsets() as offsets:
            # Offsets only: shifted, not resampled
            self.flight.sync(common_time=False, drone_records=())
        rows = offset_rows(offsets)
        section.add(
            "Offsets to the payload GPS time (inclinometer and camera: to Litchi).",
            "",
            md_table(rows, digits=3),
        )
        self.tables["sync_offsets"] = pl.DataFrame(
            [{k: v for k, v in r.items()} for r in rows],
            schema={
                "source": pl.String,
                "offset_s": pl.Float64,
                "correlation": pl.Float64,
                "method": pl.String,
                "spatial_offset_m": pl.Float64,
            },
        )
        self.warnings += offset_warnings(offsets)
        if "drone_gps" not in offsets:
            self.warnings.append("Sync: no drone GPS offset found")

    # GNSS -------------------------------------------------------------------------

    def _convbin(self, raw: Path, obs: Path, nav: Path, span=None) -> bool:
        from pils.analyze.ppk_tuning import rtklib_binary

        command = [rtklib_binary("convbin"), "-od", "-os", "-oi", "-ot", "-ol"]
        command += ["-r", "ubx"]
        if span is not None:
            for flag, moment in (("-ts", span[0]), ("-te", span[1])):
                command += [flag, *moment.strftime("%Y/%m/%d %H:%M:%S").split()]
        command += ["-o", str(obs), "-n", str(nav), str(raw)]
        subprocess.run(command, check=True, capture_output=True)
        return obs.exists() and obs.stat().st_size > 0

    def _base_candidates(self) -> list[Path]:
        """Raw base recordings: the day's, then the campaign's (as PPK)."""
        day = self.flight_path.parent / "base"
        campaign = self.flight_path.parent.parent / "base"
        files = sorted(p for p in day.glob("*") if p.suffix.lower() == ".ubx")
        if campaign.is_dir():
            files += sorted(
                p
                for p in campaign.rglob("*")
                if p.is_file()
                and (
                    p.suffix.lower() == ".ubx"
                    or p.name.startswith("rtk_data_")
                    or (p.suffix == ".bin" and "ZED-F9P" in p.name)
                )
            )
        return files

    def _gnss_section(
        self, section: Section, rinex: bool, ppk_template: str | Path | None
    ) -> None:
        from pils.analyze.ppk_tuning import (
            data_quality_note,
            profile_obs,
            rtklib_binary,
        )

        sensors = Path(self.flight_info["aux_data_folder_path"]) / "sensors"
        rovers = sorted([*sensors.glob("*_GPS.bin"), *sensors.glob("*_GPS.ubx")])
        if not rinex:
            section.add("Not done (rinex=False).")
            return
        if not rovers:
            section.add(f"No rover recording (*_GPS.bin/.ubx) in {sensors}.")
            self.warnings.append("GNSS: no rover recording, no PPK possible")
            return
        convbin = rtklib_binary("convbin")
        if shutil.which(convbin) is None:
            section.add(
                "RTKLIB convbin not found (set PILS_RTKLIB_BIN or add it to PATH): "
                "rover and base not profiled."
            )
            return
        with tempfile.TemporaryDirectory(prefix="pils_quicklook_") as scratch:
            work = Path(scratch)
            rover_obs, rover_nav = work / "rover.obs", work / "rover.nav"
            if not self._convbin(rovers[0], rover_obs, rover_nav):
                section.add(f"convbin made no observations from {rovers[0].name}.")
                self.warnings.append("GNSS: rover recording has no observations")
                return
            rover_profile = profile_obs(rover_obs)
            rover_data = read_rinex(rover_obs)
            base_obs, base_nav = work / "base.obs", work / "base.nav"
            base_raw = self._find_base(rover_profile, base_obs, base_nav)
            base_data = read_rinex(base_obs) if base_raw else None
            base_profile = profile_obs(base_obs) if base_raw else None

            rows = []
            for label, data, profile in (
                ("rover", rover_data, rover_profile),
                ("base", base_data, base_profile),
            ):
                if data is None or profile is None:
                    continue
                for row in band_table(data[1], profile.interval_s):
                    rows.append({"receiver": label, **row})
            span = f"{rover_profile.start} - {rover_profile.end}"
            section.add(
                f"Rover: {rovers[0].name}, {rover_profile.epochs} epochs at "
                f"{rover_profile.interval_s:g} s, {span}.",
            )
            if base_raw is None:
                section.add(
                    "",
                    "No base recording covers the rover (looked in the day's and "
                    "the campaign's base/ folders): no PPK.",
                )
                self.warnings.append("GNSS: no base recording covers the flight")
            else:
                section.add("", f"Base: {base_raw.name} ({base_raw.parent}).")
            section.add(
                "",
                "Median per epoch: satellites tracked, and above 30/35/40 dB-Hz; "
                "SNR median and 10th percentile; loss of lock on the band's "
                "phase; continuous tracking (observation-weighted arc length).",
                "",
                md_table(rows),
            )
            if rows:
                self.tables["gnss_bands"] = pl.DataFrame(rows)

            summary: list[dict[str, Any]] = []
            if base_data is not None:
                common = common_satellites(rover_data, base_data)
                summary.append(
                    {
                        "": "Common satellites (median)",
                        "value": ", ".join(f"{s} {n:g}" for s, n in common.items()),
                    }
                )
                baseline = self._baseline_km(rover_profile, base_profile)
                if baseline is not None:
                    summary.append({"": "Baseline (km)", "value": f"{baseline:.2f}"})
                    self.metadata["baseline_km"] = baseline
            summary += [
                {
                    "": "Rover L1 loss of lock (%)",
                    "value": f"{100 * rover_profile.lli_fraction:.1f}",
                },
                {
                    "": "Rover L1 continuous tracking (s)",
                    "value": f"{rover_profile.arc_s:.0f}",
                },
            ]
            section.add("", md_table(summary))
            note = data_quality_note(rover_profile)
            self.metadata["rover_profile"] = rover_profile.as_dict()
            if note:
                section.add("", f"**{note}** (any configuration).")
                self.warnings.append(f"GNSS: {note}")
            else:
                section.add("", "Rover loss of lock and tracking length look fine.")
            self._plot_gnss(rover_data)
            self._link(section, "gnss_rover", "Rover satellites")

            if ppk_template is not None:
                if base_raw is None:
                    section.add("", "PPK tuning not run: no base.")
                else:
                    nav = base_nav if base_nav.exists() else rover_nav
                    self._ppk(section, ppk_template, rover_obs, base_obs, nav, work)

    def _find_base(self, rover, obs: Path, nav: Path) -> Path | None:
        """Convert the first base recording that covers the rover (+-10 min)."""
        from pils.analyze.ppk_tuning import profile_obs

        if rover.start is None or rover.end is None:
            return None
        start = rover.start.astype(datetime.datetime)
        end = rover.end.astype(datetime.datetime)
        span = (
            start - datetime.timedelta(minutes=10),
            end + datetime.timedelta(minutes=10),
        )
        for raw in self._base_candidates():
            try:
                if not self._convbin(raw, obs, nav, span):
                    continue
            except subprocess.CalledProcessError:
                continue
            base = profile_obs(obs)
            if (
                base.start is not None
                and base.end is not None
                and base.start <= rover.start + np.timedelta64(2, "s")
                and base.end >= rover.end - np.timedelta64(2, "s")
            ):
                return raw
        return None

    def _baseline_km(self, rover, base) -> float | None:
        """Rover (payload GPS median ECEF, else RINEX header) to base."""
        if base is None or base.approx_xyz is None:
            return None
        rover_xyz = rover.approx_xyz
        gps = getattr(self.flight.raw_data.payload_data, "gps", None)
        if isinstance(gps, pl.DataFrame) and "ecef_x_m" in gps.columns:
            good = gps.filter(
                pl.col("ecef_x_m").is_not_null() & (pl.col("ecef_x_m") != 0)
            )
            if good.height:
                rover_xyz = tuple(
                    float(np.median(good[c].to_numpy()))
                    for c in ("ecef_x_m", "ecef_y_m", "ecef_z_m")
                )
        if rover_xyz is None:
            return None
        return float(np.linalg.norm(np.subtract(rover_xyz, base.approx_xyz)) / 1e3)

    def _reference_track(self) -> pl.DataFrame | None:
        drone = self.flight.raw_data.drone_data.drone
        if not isinstance(drone, pl.DataFrame) or "rtk_lat_deg" not in drone.columns:
            return None
        track = drone.select(
            "timestamp",
            pl.col("rtk_lat_deg").alias("lat_deg"),
            pl.col("rtk_lon_deg").alias("lon_deg"),
            pl.col("rtk_h_ell_m").alias("h_ell_m"),
        ).filter((pl.col("lat_deg") != 0) & pl.col("timestamp").is_not_null())
        return track if track.height else None

    def _ppk(
        self,
        section: Section,
        template: str | Path,
        rover_obs: Path,
        base_obs: Path,
        nav: Path,
        work: Path,
    ) -> None:
        from pils.analyze.ppk_tuning import tune

        reference = self._reference_track()
        result = tune(template, rover_obs, base_obs, nav, work / "runs", reference)
        trials = result.table()
        self.tables["ppk_tuning"] = trials
        self.metadata["ppk_validation"] = result.validation
        section.add(
            "",
            "### PPK tuning",
            "",
            f"Template {Path(str(template)).name}, {len(result.trials)} candidates, "
            f"validated on the {result.validation}.",
        )
        if result.base_note:
            section.add("", f"Base position check: {result.base_note}.")
        if any(c.overrides.get("ant2-postype") == "single" for c in result.trials):
            self.warnings.append(f"PPK: {result.base_note}")
        best = result.best
        if best is None:
            section.add("", "**No candidate passed validation**: no reliable fixes.")
            self.warnings.append("PPK: no configuration gives validated fixes")
        else:
            assert result.config_text is not None and self.revision_dir is not None
            (self.revision_dir / "optimized.conf").write_text(result.config_text)
            self.metadata["ppk_best"] = best.row()
            section.add(
                "",
                f"Best: **{best.name}**, {best.fix_pct:.1f} % fixed, "
                f"RMS {_fmt(best.h_rms, 3)} m horizontal / {_fmt(best.v_rms, 3)} m "
                f"vertical, {_fmt(best.bad_pct, 2)} % wrong fixes. "
                "Configuration: `optimized.conf` next to this report.",
                "",
                "Settings changed from the template:",
                "",
                *(f"- `{k} = {v}`" for k, v in best.overrides.items()),
            )
            if best.fix_pct < 50:
                self.warnings.append(f"PPK: best fix rate only {best.fix_pct:.0f} %")
        top = trials.sort(["valid", "fix_pct"], descending=True).head(10)
        columns = ["name", "fix_pct", "h_rms_m", "v_rms_m", "bad_pct", "valid", "note"]
        columns = [c for c in columns if c in top.columns]
        section.add(
            "", "Top candidates:", "", md_table(top.select(columns).to_dicts(), 2)
        )

    # AZEL ---------------------------------------------------------------------------

    def _azel_section(self, section: Section, telescopes: Iterable[str]) -> None:
        from pils.analyze.azel import AZELAnalysis
        from pils.config.telescopes import TELESCOPES_CONFIG

        geod: dict[str, float] = TELESCOPES_CONFIG["dji_broadcast_geod"]  # type: ignore[assignment]
        rows: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="pils_quicklook_") as scratch:
            azel = AZELAnalysis(self.flight, output_dir=scratch)
            for telescope in telescopes:
                try:
                    version = azel.run_analysis(telescope, geod, save_data=False)
                except (ValueError, FileNotFoundError) as e:
                    rows.append({"telescope": telescope, "note": str(e).split("\n")[0]})
                    continue
                if version is None:
                    rows.append({"telescope": telescope, "note": "no valid RTK data"})
                    continue
                data = version.azel_data
                if self.airborne is not None:
                    airborne = data.filter(
                        pl.col("timestamp").is_between(*self.airborne)
                    )
                    data = airborne if airborne.height else data
                rows.append(
                    {
                        "telescope": telescope,
                        "el_min_deg": float(np.min(data["el_deg"].to_numpy())),
                        "el_max_deg": float(np.max(data["el_deg"].to_numpy())),
                        "az_min_deg": float(np.min(data["az_deg"].to_numpy())),
                        "az_max_deg": float(np.max(data["az_deg"].to_numpy())),
                        "range_min_m": float(np.min(data["srange_m"].to_numpy())),
                        "range_max_m": float(np.max(data["srange_m"].to_numpy())),
                        "note": "",
                    }
                )
        if all("el_min_deg" not in r for r in rows):
            section.add("No AZEL: telescope coordinates missing for this campaign.", "")
        section.add(
            "Airborne part, not saved (`pils package` saves it).", "", md_table(rows)
        )

    # Plots ---------------------------------------------------------------------------

    def _figure(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.2), dpi=110)
        ax.grid(True, color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        return plt, fig, ax

    def _link(self, section: Section, name: str, title: str) -> None:
        if (self.plots_dir / f"{name}.png").exists():
            section.add("", f"![{title}](plots/{name}.png)")

    def _origin(self, times: np.ndarray) -> float:
        return self.airborne[0] if self.airborne else float(times.min())

    def _plot_lines(self, name: str, lines: list, ylabel: str) -> None:
        plt, fig, ax = self._figure()
        origin = self._origin(lines[0][1])
        for k, (label, times, values) in enumerate(lines):
            step = max(1, times.size // 20000)  # enough points for a PNG
            ax.plot(
                (times[::step] - origin) / 60,
                values[::step],
                color=COLORS[k % len(COLORS)],
                linewidth=1.2,
                label=label,
            )
        if self.airborne:
            ax.axvspan(0, (self.airborne[1] - origin) / 60, color="#f0efec", zorder=0)
        ax.set_xlabel("Minutes from take-off" if self.airborne else "Minutes")
        ax.set_ylabel(ylabel)
        if len(lines) > 1:
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(self.plots_dir / f"{name}.png")
        plt.close(fig)

    def _plot_track(self, east: np.ndarray, north: np.ndarray, minutes) -> None:
        plt, fig, ax = self._figure()
        fig.set_size_inches(5.5, 5)
        from matplotlib.colors import LinearSegmentedColormap

        # Light-to-dark blue, light end still visible on white
        cmap = LinearSegmentedColormap.from_list("time", ["#9ec5f4", "#0d366b"])
        points = ax.scatter(east, north, c=minutes, s=2, cmap=cmap)
        fig.colorbar(points, ax=ax, label="Minutes from take-off")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("East of take-off (m)")
        ax.set_ylabel("North of take-off (m)")
        fig.tight_layout()
        fig.savefig(self.plots_dir / "drone_track.png")
        plt.close(fig)

    def _plot_gnss(self, rover: tuple[list[str], list[dict[str, dict]]]) -> None:
        times = np.array(
            [_rinex_time(t).astype("datetime64[s]").astype(float) for t in rover[0]]
        )
        systems = sorted({sat[0] for epoch in rover[1] for sat in epoch})
        lines = []
        for system in systems:
            band = "2" if system == "C" else "1"
            counts = np.array(
                [
                    sum(
                        1
                        for sat, obs in epoch.items()
                        if sat[0] == system
                        and band in obs
                        and (obs[band][0] or 0) >= SNR_LEVELS[1]
                    )
                    for epoch in rover[1]
                ]
            )
            if counts.any():
                lines.append((f"{system}{band}", times, counts))
        if lines:
            self._plot_lines(
                "gnss_rover", lines, f"Rover satellites ≥ {SNR_LEVELS[1]} dB-Hz"
            )
