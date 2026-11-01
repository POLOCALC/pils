"""Reading RINEX observation and navigation files; satellite azimuth/elevation."""

import logging
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from ..utils import (
    get_frequency_band,
)

logger = logging.getLogger(__name__)


class RINEXParsingMixin:
    """Parsing and satellite geometry for :class:`~pils.analyze.ppkdata.RINEX.analyzer.RINEXAnalyzer`."""

    def parse_obs_file(self, snr_only: bool = False, sample_rate: int = 1):
        """Parse RINEX file into Polars DataFrame.

        Performs full-fidelity parsing of RINEX observation file,
        extracting all observation types and metadata.

        Args:
            snr_only: If True, only parse SNR observations
            sample_rate: Epoch sampling rate (1 = all epochs)

        Returns
        -------
            Polars DataFrame with parsed observations

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')
            >>> df = analyzer.parse(sample_rate=30)  # Every 30s
            >>> print(df.shape)
        """
        logger.info(f"Parsing RINEX file: {self.obsname}")
        in_header = True
        records = []
        epoch_counter = 0
        current_epoch = None  # Initialize before loop to avoid unbound errors

        with open(self.obspath) as f:
            for line in f:
                if in_header:
                    if "END OF HEADER" in line:
                        in_header = False
                    elif "SYS / # / OBS TYPES" in line:
                        sys = line[0]
                        t = line[7:60].split()
                        if sys not in self.obs_types:
                            self.obs_types[sys] = t
                        else:
                            self.obs_types[sys].extend(t)
                    elif "GLONASS SLOT / FRQ #" in line:
                        content = line[:60]
                        # First line may have a count in the first 3 chars
                        start_idx = 4 if content[:3].strip().isdigit() else 0
                        parts = content[start_idx:].split()
                        for j in range(0, len(parts), 2):
                            if j + 1 < len(parts):
                                self.glo_slots[parts[j]] = int(parts[j + 1])
                    elif "APPROX POSITION XYZ" in line:
                        parts = line[:60].split()
                        if len(parts) >= 3:
                            self.header_info["position"] = [
                                float(parts[0]),
                                float(parts[1]),
                                float(parts[2]),
                            ]
                    continue

                if line.startswith(">"):
                    try:
                        p = line.split()
                        dt = datetime(
                            int(p[1]),
                            int(p[2]),
                            int(p[3]),
                            int(p[4]),
                            int(p[5]),
                            int(float(p[6])),
                        )
                        dt += timedelta(seconds=float(p[6]) % 1)
                        epoch_counter += 1
                        current_epoch = dt if epoch_counter % sample_rate == 0 else None
                        if current_epoch:
                            self.epochs.append(dt)
                    except (ValueError, IndexError):
                        current_epoch = None
                    continue

                if current_epoch is None:
                    continue
                if len(line) < 4:
                    continue
                sat_id = line[0:3].strip()
                const = sat_id[0]
                if const not in self.obs_types:
                    continue

                obs_line = line[3:]
                obs_list = self.obs_types[const]
                for i, obs_type in enumerate(obs_list):
                    if snr_only and not obs_type.startswith("S"):
                        continue
                    start = i * 16
                    if start >= len(obs_line):
                        break
                    val_str = obs_line[start : start + 14].strip()
                    if not val_str:
                        continue

                    # LLI is the character at index 14 of the 16-char block
                    lli_str = obs_line[start + 14 : start + 15].strip()
                    lli = int(lli_str) if lli_str else 0

                    try:
                        if current_epoch is not None:
                            records.append(
                                {
                                    "time": current_epoch,
                                    "satellite": sat_id,
                                    "constellation": const,
                                    "frequency": get_frequency_band(const, obs_type[1]),
                                    "obs_type": obs_type[0],
                                    "value": float(val_str),
                                    "lli": lli,
                                }
                            )
                    except ValueError:
                        pass

        self.df_obs = pl.DataFrame(records)
        logger.info(
            f"Parsed {len(self.df_obs)} observations across {len(self.epochs)} epochs"
        )

    def parse_nav_file(self):
        """Robust RINEX 3 NAV parser for GPS, Galileo, BeiDou, and GLONASS."""
        self.nav_data = {}
        if not self.navpath.exists():
            logger.warning(f"NAV file not found: {self.navpath}")
            return

        logger.info(f"Parsing NAV data from {self.navpath.name}")
        with open(self.navpath) as f:
            header_end = False
            for line in f:
                if "END OF HEADER" in line:
                    header_end = True
                    break

            if not header_end:
                return

            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i]
                if not line.strip():
                    i += 1
                    continue

                sat_id = line[0:3].strip()
                if not sat_id or len(sat_id) < 2:
                    i += 1
                    continue
                const = sat_id[0]

                # Determine record length based on constellation (RINEX 3 standard)
                # GPS/GAL/BDS usually 8-line records (7 data lines)
                # GLO usually 4-line records (3 data lines)
                n_data_lines = 3 if const == "R" else 7

                try:
                    # Collect all lines for this record
                    record_lines = lines[i : i + 1 + n_data_lines]
                    data_str = ""
                    # Header line (line 0) has epoch and first 3 values
                    data_str += record_lines[0][23:].replace("D", "E").replace("d", "e")
                    # Data lines (1 to n)
                    for dl in record_lines[1:]:
                        data_str += dl.replace("D", "E").replace("d", "e")

                    vals = [float(v) for v in data_str.split()]

                    # Parse epoch from header line
                    # Format: YYYY MM DD HH mm ss
                    # Note: RINEX 3 NAV header lines start with SatID, then YYYY...
                    ep_str = record_lines[0][3:23]
                    parts = ep_str.split()
                    epoch = datetime(
                        int(parts[0]),
                        int(parts[1]),
                        int(parts[2]),
                        int(parts[3]),
                        int(parts[4]),
                        int(float(parts[5])),
                    )

                    if const in "GEC":  # GPS, Galileo, BeiDou
                        if len(vals) < 21:
                            i += 1 + n_data_lines
                            continue
                        eph = {
                            "epoch": epoch,
                            "af0": vals[0],
                            "af1": vals[1],
                            "af2": vals[2],
                            "iode": vals[3],
                            "crs": vals[4],
                            "delta_n": vals[5],
                            "M0": vals[6],
                            "cuc": vals[7],
                            "e": vals[8],
                            "cus": vals[9],
                            "sqrt_a": vals[10],
                            "toe": vals[11],
                            "cic": vals[12],
                            "omega0": vals[13],
                            "cis": vals[14],
                            "i0": vals[15],
                            "crc": vals[16],
                            "omega": vals[17],
                            "omega_dot": vals[18],
                            "i_dot": vals[19],
                        }
                    elif const == "R":  # GLONASS
                        if len(vals) < 12:
                            i += 1 + n_data_lines
                            continue
                        eph = {
                            "epoch": epoch,
                            "tau_n": vals[0],
                            "gamma_n": vals[1],
                            "tk": vals[2],
                            "x": vals[3] * 1000.0,
                            "vx": vals[4] * 1000.0,
                            "ax": vals[5] * 1000.0,
                            "health": vals[6],
                            "y": vals[7] * 1000.0,
                            "vy": vals[8] * 1000.0,
                            "ay": vals[9] * 1000.0,
                            "freq_num": vals[10],
                            "z": vals[11] * 1000.0,
                            "vz": vals[12] * 1000.0,
                            "az": vals[13] * 1000.0,
                            "age": vals[14],
                        }
                    else:
                        i += 1 + n_data_lines
                        continue

                    if sat_id not in self.nav_data:
                        self.nav_data[sat_id] = []
                    self.nav_data[sat_id].append(eph)

                except Exception:
                    pass

                i += 1 + n_data_lines

    def compute_satellite_azel(self):
        """Propagates satellite positions for precise Az/El calculation.

        Uses broadcast ephemeris to calculate satellite positions and
        compute azimuth/elevation angles from receiver position.

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs', navpath='file.nav')
            >>> analyzer.parse_obs_file()
            >>> analyzer.parse_nav_file()
            >>> analyzer.compute_satellite_azel()
            >>> print(analyzer.azel_df.head())
        """
        if not hasattr(self, "nav_data") or not self.nav_data:
            logger.warning("No NAV data loaded. Falling back to mock geometry")
            return self._mock_azel()

        logger.info("Computing precise Az/El from NAV ephemeris")
        receiver_pos = self.header_info.get("position")
        if not receiver_pos:
            logger.warning("Receiver position unknown. Mocking Az/El")
            return self._mock_azel()

        # Constants
        GM = 3.986005e14
        OMEGA_E = 7.2921151467e-5

        azel_list = []
        for sat in self.df_obs["satellite"].unique():
            if sat not in self.nav_data:
                continue

            # Sort ephemeris by epoch
            eph_list = sorted(self.nav_data[sat], key=lambda x: x["epoch"])

            for t in self.epochs:
                # Find closest ephemeris (within 4 hours)
                closest = min(
                    eph_list, key=lambda e: abs((e["epoch"] - t).total_seconds())
                )
                dt = (t - closest["epoch"]).total_seconds()
                if abs(dt) > 14400:
                    continue

                try:
                    if sat[0] in "GEC":
                        # Keplerian Propagation
                        e = closest["e"]
                        a = closest["sqrt_a"] ** 2
                        n = np.sqrt(GM / a**3) + closest["delta_n"]
                        M = closest["M0"] + n * dt

                        E = M
                        for _ in range(10):
                            E = M + e * np.sin(E)

                        v = 2 * np.arctan2(
                            np.sqrt(1 + e) * np.sin(E / 2),
                            np.sqrt(1 - e) * np.cos(E / 2),
                        )
                        phi = v + closest["omega"]

                        du = closest["cus"] * np.sin(2 * phi) + closest["cuc"] * np.cos(
                            2 * phi
                        )
                        dr = closest["crs"] * np.sin(2 * phi) + closest["crc"] * np.cos(
                            2 * phi
                        )
                        di = closest["cis"] * np.sin(2 * phi) + closest["cic"] * np.cos(
                            2 * phi
                        )

                        u = phi + du
                        r = a * (1 - e * np.cos(E)) + dr
                        i = closest["i0"] + di + closest["i_dot"] * dt

                        x_op, y_op = r * np.cos(u), r * np.sin(u)
                        omega = (
                            closest["omega0"]
                            + (closest["omega_dot"] - OMEGA_E) * dt
                            - OMEGA_E * closest["toe"]
                        )

                        x, y, z = (
                            x_op * np.cos(omega) - y_op * np.cos(i) * np.sin(omega),
                            x_op * np.sin(omega) + y_op * np.cos(i) * np.cos(omega),
                            y_op * np.sin(i),
                        )
                    else:  # GLONASS Simplified Linear
                        x = (
                            closest["x"]
                            + closest["vx"] * dt
                            + 0.5 * closest["ax"] * dt**2
                        )
                        y = (
                            closest["y"]
                            + closest["vy"] * dt
                            + 0.5 * closest["ay"] * dt**2
                        )
                        z = (
                            closest["z"]
                            + closest["vz"] * dt
                            + 0.5 * closest["az"] * dt**2
                        )
                        # Earth rotation correction for GLO
                        angle = OMEGA_E * dt
                        x_rot = x * np.cos(angle) + y * np.sin(angle)
                        y_rot = -x * np.sin(angle) + y * np.cos(angle)
                        x, y = x_rot, y_rot

                    # ENU Conversion
                    dx, dy, dz = (
                        x - receiver_pos[0],
                        y - receiver_pos[1],
                        z - receiver_pos[2],
                    )
                    p = np.sqrt(receiver_pos[0] ** 2 + receiver_pos[1] ** 2)
                    lon = np.arctan2(receiver_pos[1], receiver_pos[0])
                    lat = np.arctan2(receiver_pos[2], p * (1 - 0.00669437999))

                    e_enu = -np.sin(lon) * dx + np.cos(lon) * dy
                    n_enu = (
                        -np.sin(lat) * np.cos(lon) * dx
                        - np.sin(lat) * np.sin(lon) * dy
                        + np.cos(lat) * dz
                    )
                    u_val = (
                        np.cos(lat) * np.cos(lon) * dx
                        + np.cos(lat) * np.sin(lon) * dy
                        + np.sin(lat) * dz
                    )

                    az = np.rad2deg(np.arctan2(e_enu, n_enu)) % 360
                    el = np.rad2deg(np.arctan2(u_val, np.sqrt(e_enu**2 + n_enu**2)))
                    azel_list.append(
                        {"time": t, "satellite": sat, "azimuth": az, "elevation": el}
                    )
                except Exception:
                    pass

        self.azel_df = pl.DataFrame(azel_list)

    def _mock_azel(self):
        """Generate mock azimuth/elevation data when navigation unavailable.

        Creates simulated satellite tracks for visualization when
        precise ephemeris is not available.

        Examples
        --------
            >>> analyzer = RINEXAnalyzer('file.obs')  # No nav file
            >>> analyzer.parse_obs_file()
            >>> analyzer._mock_azel()
            >>> print(f"Mock tracks for {analyzer.azel_df['satellite'].n_unique()} satellites")
        """
        """Unified mock logic if NAV is missing."""
        azel_list = []
        for sat in self.df_obs["satellite"].unique():
            seed = sum(ord(c) for c in sat)
            start_az, start_el = (seed * 137.5) % 360, 20 + (seed * 17.3) % 50
            for t in self.epochs:
                dt = (t - self.epochs[0]).total_seconds()
                az = (start_az + 0.005 * dt) % 360
                el = start_el + 0.002 * dt
                if el < 0:
                    el = -el
                if el > 90:
                    el = 180 - el
                azel_list.append(
                    {"time": t, "satellite": sat, "azimuth": az, "elevation": el}
                )
        self.azel_df = pl.DataFrame(azel_list)
