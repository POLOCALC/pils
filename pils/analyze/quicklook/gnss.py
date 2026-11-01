"""GNSS sections of the quick look: rover and base RINEX, PPK, telescope AZEL."""

from __future__ import annotations

import datetime
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pils.analyze.quicklook.common import (
    SNR_LEVELS,
    Section,
    _fmt,
    md_table,
)
from pils.analyze.quicklook.rinex import (
    _rinex_time,
    band_table,
    common_satellites,
    read_rinex,
)
from pils.utils.logging_config import get_logger

logger = get_logger(__name__)


class QuickLookGNSSMixin:
    """The GNSS, PPK and AZEL sections of :class:`~pils.analyze.quicklook.QuickLook`."""

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

        result = tune(template, rover_obs, base_obs, nav, work / "runs")
        trials = result.table()
        self.tables["ppk_tuning"] = trials
        self.metadata["ppk_validation"] = result.validation
        section.add(
            "",
            "### PPK tuning",
            "",
            f"Template {Path(str(template)).name}, {len(result.trials)} candidates, "
            f"chosen by {result.validation}.",
        )
        if result.base_note:
            section.add("", f"Base position check: {result.base_note}.")
        if any(c.overrides.get("ant2-postype") == "single" for c in result.trials):
            self.warnings.append(f"PPK: {result.base_note}")
        best = result.best
        if best is None:
            section.add("", "**No candidate fixes any epoch.**")
            self.warnings.append("PPK: no configuration fixes any epoch")
        else:
            assert result.config_text is not None and self.revision_dir is not None
            (self.revision_dir / "optimized.conf").write_text(result.config_text)
            self.metadata["ppk_best"] = best.row()
            quality = (
                f"{_fmt(best.good_pct, 1)} % good fixes, "
                f"{_fmt(best.wrong_pct, 2)} % wrong fixes (of the epochs)"
                if best.good_pct is not None
                else f"RMS {_fmt(best.h_rms, 3)} m horizontal / "
                f"{_fmt(best.v_rms, 3)} m vertical, {_fmt(best.bad_pct, 2)} % "
                "wrong fixes"
            )
            section.add(
                "",
                f"Best: **{best.name}**, {best.fix_pct:.1f} % fixed, {quality}. "
                "Configuration: `optimized.conf` next to this report.",
                "",
                "Settings changed from the template:",
                "",
                *(f"- `{k} = {v}`" for k, v in best.overrides.items()),
            )
            if best.fix_pct < 50:
                self.warnings.append(f"PPK: best fix rate only {best.fix_pct:.0f} %")
        top = trials.head(10)  # sorted by tune: valid first, then the best
        columns = [
            "name",
            "fix_pct",
            "good_pct",
            "wrong_pct",
            "h_rms_m",
            "v_rms_m",
            "bad_pct",
            "valid",
            "note",
        ]
        columns = [
            c for c in columns if c in top.columns and top[c].null_count() < top.height
        ]
        section.add(
            "", "Top candidates:", "", md_table(top.select(columns).to_dicts(), 2)
        )
        effects = result.effects
        if effects:
            section.add(
                "",
                "How the settings acted on this flight (random configurations; "
                "gain: best good fixes with minus without, risk: wrong fixes "
                "with minus without, % of epochs):",
                "",
                md_table(
                    [{k: e[k] for k in ("choice", "gain", "risk")} for e in effects], 1
                ),
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
                    version = azel.run_analysis(
                        telescope,
                        geod,
                        save_data=False,
                        telescope_source=getattr(self, "telescope_source", "survey"),
                    )
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
                meta = version.metadata
                note = f"from {meta.get('telescope_position_source')}"
                if not meta.get("rtk_applied"):
                    note += "; no DJI base correction"
                    self.warnings.append(
                        f"AZEL {telescope}: no surveyed DJI base in this campaign, "
                        "the drone track is not corrected for it"
                    )
                if str(meta.get("telescope_position_source", "")).startswith("so3g"):
                    self.warnings.append(
                        f"AZEL {telescope}: so3g telescope position (2-3 m off)"
                    )
                rows.append(
                    {
                        "telescope": telescope,
                        "el_min_deg": float(np.min(data["el_deg"].to_numpy())),
                        "el_max_deg": float(np.max(data["el_deg"].to_numpy())),
                        "az_min_deg": float(np.min(data["az_deg"].to_numpy())),
                        "az_max_deg": float(np.max(data["az_deg"].to_numpy())),
                        "range_min_m": float(np.min(data["srange_m"].to_numpy())),
                        "range_max_m": float(np.max(data["srange_m"].to_numpy())),
                        "note": note,
                    }
                )
        if all("el_min_deg" not in r for r in rows):
            section.add("No AZEL: no telescope position (see the notes).", "")
        section.add(
            "Airborne part, not saved (`pils package` saves it).", "", md_table(rows)
        )

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
