"""QuickLook: the quick-look report of one flight (see :mod:`pils.analyze.quicklook`)."""

from __future__ import annotations

import contextlib
import datetime
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils import store
from pils.analyze.package import DEFAULT_TELESCOPES, PAYLOAD_SENSORS, resolve_flight
from pils.analyze.quicklook.common import (
    COLORS,
    MIN_COVERAGE_PCT,
    MIN_LITCHI_OVERLAP_PCT,
    QUICKLOOK_DIR,
    REPORT_FILE,
    Section,
    _seconds,
    _size,
    _utc,
    airborne_interval,
    local_en,
    md_table,
    offset_rows,
    offset_warnings,
    overlap_pct,
    rtk_flag_summary,
    table_summary,
)
from pils.analyze.quicklook.gnss import QuickLookGNSSMixin
from pils.integrations.stout import layout
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


# -- the report ----------------------------------------------------------------


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


class QuickLook(QuickLookGNSSMixin):
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
        self.quicklook_dir = base / layout.PROC_FOLDER / QUICKLOOK_DIR
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
        telescope_source: str = "survey",
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
            (:func:`pils.analyze.ppk_tuning.tune`, the most fixed epochs)
            and report the result. Takes a few minutes.
        telescope_source : {"survey", "so3g"}, default="survey"
            Telescope positions for AZEL (see
            ``pils.sensors.emlid.reference_positions``).

        Returns
        -------
        Path
            The report (``report.md``).
        """
        self.telescope_source = telescope_source
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
        """Return the report as Markdown."""
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
                    stout_takeoff = stout_takeoff.replace(tzinfo=datetime.UTC)
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
        try:
            # Offsets only: shifted, not resampled
            self.flight.sync(common_time=False, drone_records=())
        except ValueError as e:
            section.add(f"No sync: {e}")
            self.warnings.append(f"Sync: not done ({e})")
            return
        offsets = self.flight.sync_offset_details
        rows = offset_rows(offsets)
        if self.flight.sync_reference == "drone_gps":
            reference = "the drone GPS time (no payload GPS covering the flight)"
            self.warnings.append(
                "Sync: the drone GPS is the reference (no payload GPS covering "
                "the flight); payload sensors keep the payload computer's time"
            )
        else:
            reference = "the payload GPS time"
        section.add(
            f"Offsets to {reference} (inclinometer and camera: to Litchi).",
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
