"""RTKLIB Runner for PPK processing."""

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class RTKLIBRunner:
    """RTKLIB PPK processing runner.
    
    Manages RTKLIB rnx2rtkp execution for post-processed kinematic positioning.
    Validates time overlap between rover and base observations before processing.
    
    Attributes
    ----------
    bin : str
        Path to rnx2rtkp binary
    output_dif : str
        Output directory path
    
    Examples
    --------
    >>> runner = RTKLIBRunner(data_path='./output', rnx2rtkp_path='rnx2rtkp')
    >>> if runner.check_overlap('rover.obs', 'base.obs'):
    ...     runner.run_ppk('rover.obs', 'base.obs', 'rover.nav')
    """
    
    def __init__(self, data_path: str, rnx2rtkp_path: str = "rnx2rtkp") -> None:
        """Initialize RTKLIB runner.
        
        Args:
            data_path: Output directory for results
            rnx2rtkp_path: Path to rnx2rtkp binary executable
        
        Examples:
            >>> runner = RTKLIBRunner('./ppk_output')
            >>> runner = RTKLIBRunner('./results', rnx2rtkp_path='/usr/local/bin/rnx2rtkp')
        """
        self.bin = rnx2rtkp_path
        self.output_dif = data_path
        data_path_obj = Path(data_path)
        data_path_obj.mkdir(parents=True, exist_ok=True)

    def _parse_rinex_epoch_line(self, line):
        """Helper to parse a RINEX 3 epoch line (> Y M D h m s).
        
        Args:
            line: RINEX epoch line string
        
        Returns:
            datetime object or None if parsing fails
        
        Examples:
            >>> runner = RTKLIBRunner('./output')
            >>> dt = runner._parse_rinex_epoch_line('> 2026 01 21 14 00 00.0000000')
            >>> print(dt)
            2026-01-21 14:00:00
        """
        try:
            parts = line.strip().split()
            # Format: > 2026 01 21 14 00 00.0000000
            y, m, d, h, mn = map(int, parts[1:6])
            s = float(parts[6])
            return datetime(y, m, d, h, mn, int(s))
        except (ValueError, IndexError):
            return None

    def _get_rinex_bounds(self, rinex_file):
        """Reads the FIRST and LAST observation timestamps.
        
        Args:
            rinex_file: Path to RINEX observation file
        
        Returns:
            Tuple of (start_datetime, end_datetime)
        
        Examples:
            >>> runner = RTKLIBRunner('./output')
            >>> start, end = runner._get_rinex_bounds('rover.obs')
            >>> duration = (end - start).total_seconds()
            >>> print(f"Data span: {duration/3600:.1f} hours")
        """
        start_dt = None
        last_dt = None

        # Read file efficiently
        with open(rinex_file, "r") as f:
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

    def check_overlap(self, rover_obs, base_obs):
        """Check time overlap between rover and base observations.
        
        Validates that rover and base station data have sufficient time overlap
        for effective PPK processing. Warns if overlap is less than 10 minutes.
        
        Args:
            rover_obs: Path to rover RINEX observation file
            base_obs: Path to base RINEX observation file
        
        Returns:
            True if adequate overlap exists, False otherwise
        
        Examples:
            >>> runner = RTKLIBRunner('./output')
            >>> if runner.check_overlap('rover.obs', 'base.obs'):
            ...     print("Time overlap OK - proceeding with PPK")
            >>> # Output:
            >>> # Rover: 2026-01-21 14:00:00 --> 2026-01-21 15:30:00
            >>> # Base:  2026-01-21 13:45:00 --> 2026-01-21 16:00:00
            >>> # Common window: 5400.0 seconds (90.0 min)
        """
        logger.info("=== Time Overlap Analysis ===")

        # Get Bounds
        r_start, r_end = self._get_rinex_bounds(rover_obs)
        b_start, b_end = self._get_rinex_bounds(base_obs)

        # Basic Check
        if not r_start or not r_end:
            logger.error(f"Failed to read timestamps from Rover: {rover_obs}")
            return False
        if not b_start or not b_end:
            logger.error(f"Failed to read timestamps from Base: {base_obs}")
            return False

        logger.info(f"Rover: {r_start} --> {r_end}")
        logger.info(f"Base:  {b_start} --> {b_end}")

        # Calculate Overlap
        overlap_start = max(r_start, b_start)
        overlap_end = min(r_end, b_end)

        duration = (overlap_end - overlap_start).total_seconds()

        if duration <= 0:
            logger.critical("NO OVERLAP DETECTED!")
            logger.critical("The Base data ends before the Rover starts (or vice versa)")
            logger.critical(f"Gap: {abs(duration):.1f} seconds")
            return False

        logger.info(f"Common window: {duration:.1f} seconds ({duration/60:.1f} min)")

        if duration < 600:  # Less than 10 mins
            logger.warning("Overlap is very short (<10 min). Solution may be unstable")

        return True

    def run_ppk(self, rover, base, nav):
        """Execute RTKLIB PPK processing.
        
        Runs rnx2rtkp binary with rover, base, and navigation files.
        Generates solution.pos output file.
        
        Args:
            rover: Path to rover RINEX observation file
            base: Path to base RINEX observation file
            nav: Path to navigation file
        
        Examples:
            >>> runner = RTKLIBRunner('./output', rnx2rtkp_path='rnx2rtkp')
            >>> runner.run_ppk('rover.obs', 'base.obs', 'rover.nav')
            >>> # Checks for solution.pos in output directory
        """
        logger.info("=== Full PPK Processing ===")
        out = Path(self.output_dif) / "solution.pos"
        # Standard Kinematic PPK settings
        cmd = [self.bin, "-k", "optimized.conf", "-o", str(out), rover, base, nav]

        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.debug(result.stdout)
        if result.stderr:
            logger.debug(result.stderr)

        if out.exists() and out.stat().st_size > 1000:
            logger.info(f"Solution created: {out} ({out.stat().st_size} bytes)")
        else:
            logger.error("PPK file empty")
            logger.error(f"RTKLIB Error Output:\n{result.stderr}")


if __name__ == "__main__":
    # --- CONFIG ---
    # Update these paths to test
    rover = "20251207_152740_GPS_rover.obs"
    base = "ReachBase_raw_20251207140516_base.obs"
    nav = "ReachBase_raw_20251207140516_base.nav"
    data_path = "./output"

    diag = RTKLIBRunner(
        data_path=data_path, rnx2rtkp_path="rnx2rtkp"
    )  # Ensure this binary is in PATH

    # --- EXECUTE ---
    if diag.check_overlap(rover, base):
        diag.run_ppk(rover, base, nav)
