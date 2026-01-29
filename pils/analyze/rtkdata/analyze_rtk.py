import os
import subprocess
import shutil
from datetime import datetime

class PPKAnalysis:
    def __init__(self, data_path, rnx2rtkp_path="rnx2rtkp"):
        self.bin = rnx2rtkp_path
        if not os.path.exists(data_path):
            os.makedirs(data_path)

    def _parse_rinex_epoch_line(self, line):
        """Helper to parse a RINEX 3 epoch line (> Y M D h m s)."""
        try:
            parts = line.strip().split()
            # Format: > 2026 01 21 14 00 00.0000000
            y, m, d, h, mn = map(int, parts[1:6])
            s = float(parts[6])
            return datetime(y, m, d, h, mn, int(s))
        except (ValueError, IndexError):
            return None

    def _get_rinex_bounds(self, rinex_file):
        """
        Reads the FIRST and LAST observation timestamps.
        Returns tuple (start_dt, end_dt).
        """
        start_dt = None
        last_dt = None
        
        # Read file efficiently
        with open(rinex_file, 'r') as f:
            # 1. Find Start Time
            for line in f:
                if line.startswith('>'):
                    start_dt = self._parse_rinex_epoch_line(line)
                    if start_dt:
                        last_dt = start_dt # Initialize last_dt
                        break
            
            # 2. Find End Time
            # We continue reading line by line. For massive files, 
            # this takes a moment but ensures we find the true last epoch.
            for line in f:
                if line.startswith('>'):
                    dt = self._parse_rinex_epoch_line(line)
                    if dt:
                        last_dt = dt
                    
        return start_dt, last_dt

    def check_overlap(self, rover_obs, base_obs):
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
        overlap_end   = min(r_end, b_end)
        
        duration = (overlap_end - overlap_start).total_seconds()

        if duration <= 0:
            print("  [CRITICAL] NO OVERLAP DETECTED!")
            print("  The Base data ends before the Rover starts (or vice versa).")
            print("  Gap: {abs(duration):.1f} seconds")
            return False
        
        print(f"  [OK] Common Window: {duration:.1f} seconds ({duration/60:.1f} min)")
        
        if duration < 600: # Less than 10 mins
            print("  [Warning] Overlap is very short (<10 min). Solution may be unstable.")
            
        return True

    def run_ppk(self, rover, base, nav):
        print("--- 3. Full PPK Processing ---")
        out = os.path.join(self.work_dir, "solution.pos")
        # Standard Kinematic PPK settings
        cmd = [self.bin, "-k", "optimized.conf", "-o", out, rover, base, nav]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        
        if os.path.exists(out) and os.path.getsize(out) > 1000:
            print(f"  [Success] Solution Created: {out} ({os.path.getsize(out)} bytes)")
        else:
            print("  [Failure] PPK file empty.")
            print("  RTKLIB Error Output:\n{result.stderr}")

if __name__ == "__main__":
    # --- CONFIG ---
    # Update these paths to test
    rover = "20251207_152740_GPS_rover.obs"
    base  = "ReachBase_raw_20251207140516_base.obs"
    nav   = "ReachBase_raw_20251207140516_base.nav"
    
    diag = PPKDiagnose(rnx2rtkp_path="rnx2rtkp") # Ensure this binary is in PATH

    # --- EXECUTE ---
    if diag.check_overlap(rover, base):
        diag.run_ppk(rover, base, nav)
