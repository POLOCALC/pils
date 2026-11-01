"""The PPK inputs: RINEX time spans, base station candidates and conversion, overlap."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from pils.analyze.ppk_tuning import rtklib_binary

logger = logging.getLogger(__name__)


class PPKInputsMixin:
    """Rover and base inputs of :class:`~pils.analyze.ppk.PPKAnalysis`."""

    def _parse_rinex_epoch_line(self, line):
        """Parse a RINEX 3 epoch line (> Y M D h m s)."""
        try:
            parts = line.strip().split()
            # Format: > 2026 01 21 14 00 00.0000000
            y, m, d, h, mn = map(int, parts[1:6])
            s = float(parts[6])
            return datetime(y, m, d, h, mn, int(s))
        except (ValueError, IndexError):
            return None

    def _get_rinex_bounds(self, rinex_file):
        """Read the first and last observation times of a RINEX file.

        Returns tuple (start_dt, end_dt).
        """
        start_dt = None
        last_dt = None

        # Read file efficiently
        with open(rinex_file) as f:
            # 1. Find Start Time
            for line in f:
                if line.startswith(">"):
                    start_dt = self._parse_rinex_epoch_line(line)
                    if start_dt:
                        last_dt = start_dt  # Initialize last_dt
                        break

            # 2. Find End Time
            # We continue reading line by line. For massive files,
            # this takes a moment but ensures we find the true last epoch.
            for line in f:
                if line.startswith(">"):
                    dt = self._parse_rinex_epoch_line(line)
                    if dt:
                        last_dt = dt

        return start_dt, last_dt

    def _covers(self, obs: Path, start: datetime, end: datetime) -> bool:
        """Check that a RINEX file spans [start, end] (to within 2 s)."""
        first, last = self._get_rinex_bounds(obs)
        if first is None or last is None:
            return False
        early = (first - start).total_seconds()
        late = (end - last).total_seconds()
        return bool(early <= 2 and late <= 2)

    def _base_candidates(self) -> list[Path]:
        """Raw u-blox base recordings: the day's, then the campaign's."""
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

    def _convert_base(
        self, obs_path: Path, nav_path: Path, start_rover: datetime, end_rover: datetime
    ) -> None:
        """Convert the base recording covering the rover's span to RINEX.

        The span is the rover's +-10 min; the first base recording that
        covers it is used.

        Raises
        ------
        FileNotFoundError
            If no base recording covers the rover.
        """
        start = start_rover - timedelta(minutes=10)
        end = end_rover + timedelta(minutes=10)
        date_start, time_start = start.strftime("%Y/%m/%d %H:%M:%S").split(" ")
        date_end, time_end = end.strftime("%Y/%m/%d %H:%M:%S").split(" ")
        for raw in self._base_candidates():
            subprocess.run(
                [
                    rtklib_binary("convbin"),
                    "-od",
                    "-os",
                    "-oi",
                    "-ot",
                    "-ol",
                    "-r",
                    "ubx",
                    "-ts",
                    date_start,
                    time_start,
                    "-te",
                    date_end,
                    time_end,
                    "-o",
                    str(obs_path),
                    "-n",
                    str(nav_path),
                    str(raw),
                ],
                check=True,
                capture_output=True,
            )
            if obs_path.exists() and self._covers(obs_path, start_rover, end_rover):
                logger.info(f"Base from {raw}")
                return
        raise FileNotFoundError(
            f"No base recording covers {start_rover} - {end_rover} "
            f"(looked in {self.flight_path.parent / 'base'} and the campaign's base/)"
        )

    def _reference_track(self) -> pl.DataFrame | None:
        """Return the drone's RTK track (UTC), to validate tuned configurations."""
        try:
            drone = getattr(self.flight.raw_data.drone_data, "drone", None)
            if (
                not isinstance(drone, pl.DataFrame)
                or "rtk_lat_deg" not in drone.columns
            ):
                self.flight.add_drone_data()
                drone = self.flight.raw_data.drone_data.drone
            if not isinstance(drone, pl.DataFrame):
                return None
            track = drone.select(
                "timestamp",
                pl.col("rtk_lat_deg").alias("lat_deg"),
                pl.col("rtk_lon_deg").alias("lon_deg"),
                pl.col("rtk_h_ell_m").alias("h_ell_m"),
            ).filter((pl.col("lat_deg") != 0) & pl.col("timestamp").is_not_null())
            return track if track.height else None
        except Exception as e:
            # Broad: the track only validates the tuning; loading the drone
            # data can fail in many ways, and the tuning then goes without it
            logger.warning(f"No drone RTK track to validate the tuning: {e}")
            return None

    def check_overlap(self, rover_obs, base_obs):
        """Print the time overlap of the rover and base observations."""
        print("--- 1. Time Overlap Analysis ---")

        # Get Bounds
        r_start, r_end = self._get_rinex_bounds(rover_obs)
        b_start, b_end = self._get_rinex_bounds(base_obs)

        # Basic Check
        if not r_start or not r_end:
            print(f"  [Error] Failed to read timestamps from Rover: {rover_obs}")
            return False
        if not b_start or not b_end:
            print(f"  [Error] Failed to read timestamps from Base: {base_obs}")
            return False

        print(f"  Rover: {r_start}  -->  {r_end}")
        print(f"  Base:  {b_start}  -->  {b_end}")

        # Calculate Overlap
        overlap_start = max(r_start, b_start)
        overlap_end = min(r_end, b_end)

        duration = (overlap_end - overlap_start).total_seconds()

        if duration <= 0:
            print("  [CRITICAL] NO OVERLAP DETECTED!")
            print("  The Base data ends before the Rover starts (or vice versa).")
            print("  Gap: {abs(duration):.1f} seconds")
            return False

        print(f"  [OK] Common Window: {duration:.1f} seconds ({duration / 60:.1f} min)")

        if duration < 600:  # Less than 10 mins
            print(
                "  [Warning] Overlap is very short (<10 min). Solution may be unstable."
            )

        return True
