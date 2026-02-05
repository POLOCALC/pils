"""
RINEX Quality Analyzer v5 - Polars Core (Feature Complete)
High-performance GNSS analysis with 100% geodetic logic fidelity.
"""

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl

# =============================================================================
# CONSTANTS & MAPPING
# =============================================================================

C = 299792458.0  # Speed of light m/s

# Frequency in MHz
GNSS_FREQUENCIES = {
    "G": {"L1": 1575.42, "L2": 1227.60, "L5": 1176.45},
    "R": {"G1": 1602.0, "G2": 1246.0},  # Nominal, ignoring FDMA channels for MP
    "E": {
        "E1": 1575.42,
        "E5a": 1176.45,
        "E5b": 1207.14,
        "E6": 1278.75,
        "E5ab": 1191.795,
    },
    "C": {"B1": 1561.098, "B2": 1207.14, "B3": 1268.52},
    "S": {"L1": 1575.42},
    "J": {"L1": 1575.42, "L2": 1227.60, "L5": 1176.45},
}

# Compatibility names for reporting/plotting
CONSTELLATION_NAMES = {
    "G": "GPS",
    "R": "GLONASS",
    "E": "Galileo",
    "C": "BeiDou",
    "S": "SBAS",
    "J": "QZSS",
}
CONSTELLATION_COLORS = {
    "G": "#1f77b4",
    "R": "#d62728",
    "E": "#2ca02c",
    "C": "#ff7f0e",
    "S": "#9467bd",
    "J": "#8c564b",
}
RTKLIB_bands = {"single": ["L1", "G1", "E1", "B1"], "dual": ["L2", "G2", "E5b", "B2"]}


def get_frequency_band(constellation, freq_code):
    if constellation == "C":
        return {"2": "B1", "7": "B2", "6": "B3", "1": "B1C", "5": "B2a"}.get(
            freq_code, f"B{freq_code}"
        )
    if constellation == "E":
        return {"1": "E1", "5": "E5a", "7": "E5b", "8": "E5ab", "6": "E6"}.get(
            freq_code, f"E{freq_code}"
        )
    if constellation == "R":
        return {"1": "G1", "2": "G2"}.get(freq_code, f"G{freq_code}")
    return {"1": "L1", "2": "L2", "5": "L5"}.get(freq_code, f"L{freq_code}")


def get_dual_freq_bands(constellation):
    return {
        "G": ("L1", "L2"),
        "R": ("G1", "G2"),
        "E": ("E1", "E5b"),
        "C": ("B1", "B2"),
        "S": ("L1", None),
        "J": ("L1", "L2"),
    }.get(constellation, (None, None))


# =============================================================================
# RINEX ANALYZER CLASS
# =============================================================================


class RINEXAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.df = pl.DataFrame()
        self.header_info = {}
        self.obs_types = {}
        self.epochs = []
        self.azel_df = pl.DataFrame()
        self.glo_slots = {}

    def parse(self, snr_only=False, sample_rate=1):
        """Full fidelity RINEX parsing into Polars DataFrame."""
        print(f"📂 Heavy-duty Polars decomposition of {self.filename}...")
        in_header = True
        records = []
        epoch_counter = 0
        current_epoch = None  # Initialize before loop to avoid unbound errors

        with open(self.filepath, "r") as f:
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
                    except:
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
                    except:
                        pass

        self.df = pl.DataFrame(records)
        print(f"  ✓ Unified {len(self.df)} observations across {len(self.epochs)} epochs")
        return self

    def parse_nav_file(self, nav_filepath):
        """Robust RINEX 3 NAV parser for GPS, Galileo, BeiDou, and GLONASS."""
        self.nav_data = {}
        if not os.path.exists(nav_filepath):
            print(f"⚠️ NAV file not found: {nav_filepath}")
            return

        print(f"📡 Parsing NAV data from {os.path.basename(nav_filepath)}...")
        with open(nav_filepath, "r") as f:
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

                except Exception as e:
                    pass

                i += 1 + n_data_lines

    def compute_satellite_azel(self):
        """Propagates satellite positions for precise Az/El calculation."""
        if not hasattr(self, "nav_data") or not self.nav_data:
            print("⚠️ No NAV data loaded. Falling back to mock geometry.")
            return self._mock_azel()

        print("🌍 Computing precise Az/El from NAV ephemeris...")
        receiver_pos = self.header_info.get("position")
        if not receiver_pos:
            print("⚠️ Receiver position unknown. Mocking Az/El.")
            return self._mock_azel()

        # Constants
        GM = 3.986005e14
        OMEGA_E = 7.2921151467e-5

        azel_list = []
        for sat in self.df["satellite"].unique():
            if sat not in self.nav_data:
                continue

            # Sort ephemeris by epoch
            eph_list = sorted(self.nav_data[sat], key=lambda x: x["epoch"])

            for t in self.epochs:
                # Find closest ephemeris (within 4 hours)
                closest = min(eph_list, key=lambda e: abs((e["epoch"] - t).total_seconds()))
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

                        du = closest["cus"] * np.sin(2 * phi) + closest["cuc"] * np.cos(2 * phi)
                        dr = closest["crs"] * np.sin(2 * phi) + closest["crc"] * np.cos(2 * phi)
                        di = closest["cis"] * np.sin(2 * phi) + closest["cic"] * np.cos(2 * phi)

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
                        x = closest["x"] + closest["vx"] * dt + 0.5 * closest["ax"] * dt**2
                        y = closest["y"] + closest["vy"] * dt + 0.5 * closest["ay"] * dt**2
                        z = closest["z"] + closest["vz"] * dt + 0.5 * closest["az"] * dt**2
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
                    azel_list.append({"time": t, "satellite": sat, "azimuth": az, "elevation": el})
                except:
                    pass

        self.azel_df = pl.DataFrame(azel_list)

    def _mock_azel(self):
        """Unified mock logic if NAV is missing."""
        azel_list = []
        for sat in self.df["satellite"].unique():
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
                azel_list.append({"time": t, "satellite": sat, "azimuth": az, "elevation": el})
        self.azel_df = pl.DataFrame(azel_list)

    # STATISTICS
    def get_snr(self):
        return self.df.filter(pl.col("obs_type") == "S")

    def get_snr_statistics(self):
        return (
            self.get_snr()
            .group_by(["satellite", "frequency"])
            .agg(
                [
                    pl.col("value").mean().alias("mean"),
                    pl.col("value").std().alias("std"),
                    pl.col("value").count().alias("count"),
                ]
            )
        )

    def get_global_frequency_summary(self):
        snr_summary = (
            self.get_snr()
            .group_by(["constellation", "frequency"])
            .agg(
                [
                    pl.col("value").mean().alias("mean"),
                    pl.col("value").std().alias("std"),
                    pl.col("satellite").n_unique().alias("n_satellites"),
                    pl.col("value").count().alias("count"),
                ]
            )
        )
        mp_rms = self.get_multipath_rms()
        if not mp_rms.is_empty():
            mp_sum = mp_rms.group_by(["constellation", "frequency"]).agg(
                pl.col("MP_RMS").mean().alias("mean_MP_RMS")
            )
            return snr_summary.join(mp_sum, on=["constellation", "frequency"], how="left").sort(
                ["constellation", "frequency"]
            )
        return snr_summary.with_columns(pl.lit(None).alias("mean_MP_RMS")).sort(
            ["constellation", "frequency"]
        )

    # ADVANCED GEODETIC ANALYSIS
    def estimate_multipath(self):
        """
        Implements the RTKLIB multipath algorithm (Plot::updateMp).
        1. Calculate reference ionosphere I from available dual-frequency data.
        2. Compute raw MP for all frequencies.
        3. Remove mean bias per arc (separated by jumps or cycle slips).
        """
        if self.df.is_empty():
            return pl.DataFrame()

        # 1. Prepare frequency information
        # Get all phase (L) and code (P) obs
        obs = self.df.filter(pl.col("obs_type").is_in(["L", "C", "P"]))
        if obs.is_empty():
            return pl.DataFrame()

        # Pivot obs_type to columns. Ensure unique rows first.
        pivoted = obs.unique(subset=["time", "satellite", "frequency", "obs_type"]).pivot(
            on="obs_type",
            index=["time", "satellite", "constellation", "frequency"],
            values="value",
        )

        # Merge Code types (P and C)
        code_cols = [c for c in ["P", "C"] if c in pivoted.columns]
        if not code_cols:
            return pl.DataFrame()

        if len(code_cols) == 2:
            pivoted = pivoted.with_columns(pl.col("P").fill_null(pl.col("C")).alias("P_val"))
        else:
            pivoted = pivoted.with_columns(pl.col(code_cols[0]).alias("P_val"))

        if "L" not in pivoted.columns:
            return pl.DataFrame()

        # Filter rows with both P and L
        data = pivoted.filter(pl.col("P_val").is_not_null() & pl.col("L").is_not_null())
        if data.is_empty():
            return pl.DataFrame()

        # Join with GNSS frequencies to get Hz values
        freq_rows = []
        unique_sats = data.select(["satellite", "constellation"]).unique()

        for row in unique_sats.iter_rows(named=True):
            sat = row["satellite"]
            const = row["constellation"]
            if const not in GNSS_FREQUENCIES:
                continue

            for band, f_mhz in GNSS_FREQUENCIES[const].items():
                f_hz = f_mhz * 1e6
                # GLONASS FDMA adjustment
                if const == "R":
                    slot = self.glo_slots.get(sat)
                    if slot is not None:
                        if band == "G1":
                            f_hz = (1602.0 + slot * 0.5625) * 1e6
                        elif band == "G2":
                            f_hz = (1246.0 + slot * 0.4375) * 1e6

                freq_rows.append(
                    {
                        "satellite": sat,
                        "constellation": const,
                        "frequency": band,
                        "f_hz": f_hz,
                    }
                )

        freq_df = pl.DataFrame(freq_rows)

        data = data.join(freq_df, on=["satellite", "constellation", "frequency"])

        # Phase L in meters
        data = data.with_columns((pl.col("L") * (C / pl.col("f_hz"))).alias("L_m"))

        # 2. Reference Ionosphere I per (time, satellite)
        # Group by (time, sat) and pick first two available frequencies for the reference ionosphere
        dual = (
            data.sort(
                ["time", "satellite", "f_hz"], descending=[False, False, True]
            )  # Sort to pick primary bands
            .group_by(["time", "satellite"])
            .agg(
                [
                    pl.col("f_hz").head(2).alias("f_pair"),
                    pl.col("L_m").head(2).alias("L_pair"),
                ]
            )
            .filter(pl.col("f_pair").list.len() >= 2)
        )

        if dual.is_empty():
            return pl.DataFrame()

        dual = dual.with_columns(
            [
                pl.col("f_pair").list.get(0).alias("f1"),
                pl.col("f_pair").list.get(1).alias("f2"),
                pl.col("L_pair").list.get(0).alias("L1_m"),
                pl.col("L_pair").list.get(1).alias("L2_m"),
            ]
        )

        # Reference I at f1: I1 = (L1 - L2) / ((f1/f2)**2 - 1)
        dual = dual.with_columns(
            ((pl.col("L1_m") - pl.col("L2_m")) / ((pl.col("f1") / pl.col("f2")) ** 2 - 1)).alias(
                "I1"
            )
        )

        # Join back to calculate raw MP for ALL frequencies in that epoch
        res = data.join(dual.select(["time", "satellite", "f1", "I1"]), on=["time", "satellite"])

        # Raw MP_j = P_j - L_j - 2 * (f1/f_j)^2 * I1
        res = res.with_columns(
            (
                pl.col("P_val")
                - pl.col("L_m")
                - 2 * (pl.col("f1") / pl.col("f_hz")) ** 2 * pl.col("I1")
            ).alias("MP_raw")
        )

        # 3. Bias Removal per Arc (Sequence of continuous observations)
        res = res.sort(["satellite", "frequency", "time"])

        # Break arcs on: Time gap (> 60s) OR Raw MP jump (> 5.0m)
        res = res.with_columns(
            [
                pl.col("time")
                .diff()
                .dt.total_seconds()
                .over(["satellite", "frequency"])
                .fill_null(0)
                .alias("dt"),
                pl.col("MP_raw")
                .diff()
                .abs()
                .over(["satellite", "frequency"])
                .fill_null(0)
                .alias("jump"),
            ]
        )

        res = res.with_columns(((pl.col("dt") > 60) | (pl.col("jump") > 5.0)).alias("is_break"))

        res = res.with_columns(
            pl.col("is_break")
            .cast(pl.Int32)
            .cum_sum()
            .over(["satellite", "frequency"])
            .alias("arc_id")
        )

        # Final MP: Subtract mean per arc to remove ambiguity bias
        res = res.with_columns(
            (
                pl.col("MP_raw")
                - pl.col("MP_raw").mean().over(["satellite", "frequency", "arc_id"])
            ).alias("MP")
        )

        return res.select(["time", "satellite", "constellation", "frequency", "MP"])

    def get_multipath_rms(self):
        mp = self.estimate_multipath()
        if mp.is_empty():
            return pl.DataFrame()
        return mp.group_by(["satellite", "constellation", "frequency"]).agg(
            (pl.col("MP") ** 2).mean().sqrt().alias("MP_RMS")
        )

    def detect_cycle_slips(self, threshold_gf=0.08, threshold_mw=2.5):
        """Restores dual-combination (GF + MW) Slip Detection."""
        slips = []
        for sat in self.df["satellite"].unique().to_list():
            const = sat[0]
            b1, b2 = get_dual_freq_bands(const)
            if not b2 or const not in GNSS_FREQUENCIES:
                continue

            f1 = GNSS_FREQUENCIES[const][b1]
            f2 = GNSS_FREQUENCIES[const][b2]
            l1_wl = C / (f1 * 1e6)
            l2_wl = C / (f2 * 1e6)
            wl_wl = C / ((f1 - f2) * 1e6)

            sub = self.df.filter(pl.col("satellite") == sat)

            def get_t(kind, band):
                return sub.filter(
                    (pl.col("obs_type") == kind) & (pl.col("frequency") == band)
                ).select(["time", "value"])

            c1, c2, l1, l2 = (
                get_t("C", b1),
                get_t("C", b2),
                get_t("L", b1),
                get_t("L", b2),
            )
            if any(d.is_empty() for d in [c1, c2, l1, l2]):
                continue

            comb = (
                l1.rename({"value": "L1_cyc"})
                .join(l2.rename({"value": "L2_cyc"}), on="time")
                .join(c1.rename({"value": "C1"}), on="time")
                .join(c2.rename({"value": "C2"}), on="time")
            )

            # GF in meters
            comb = comb.with_columns(
                (pl.col("L1_cyc") * l1_wl - pl.col("L2_cyc") * l2_wl).alias("GF")
            )
            # MW in cycles
            comb = comb.with_columns(
                (
                    (
                        (f1 * pl.col("L1_cyc") * l1_wl - f2 * pl.col("L2_cyc") * l2_wl) / (f1 - f2)
                        - (f1 * pl.col("C1") + f2 * pl.col("C2")) / (f1 + f2)
                    )
                    / wl_wl
                ).alias("MW")
            )

            # Jump detection
            comb = comb.sort("time").with_columns(
                [
                    pl.col("GF").diff().abs().alias("jump_gf"),
                    pl.col("MW").diff().abs().alias("jump_mw"),
                ]
            )

            hit = comb.filter(
                (pl.col("jump_gf") > threshold_gf) | (pl.col("jump_mw") > threshold_mw)
            )
            if not hit.is_empty():
                hit = hit.with_columns(
                    (
                        pl.when(pl.col("jump_gf") > threshold_gf)
                        .then(pl.lit("GF"))
                        .otherwise(pl.lit(""))
                        + pl.when(pl.col("jump_mw") > threshold_mw)
                        .then(pl.lit("MW"))
                        .otherwise(pl.lit(""))
                    ).alias("type")
                )
                slips.append(
                    hit.select(["time", "type"]).with_columns(pl.lit(sat).alias("satellite"))
                )

        return pl.concat(slips) if slips else pl.DataFrame()

    def get_completeness_metrics(self):
        """Calculates observation rate vs expected capacity (Obs / (Sats * Epochs * 2))."""
        if not self.epochs or self.df.is_empty():
            return 0.0
        n_epochs = len(self.epochs)
        n_sats = self.df["satellite"].n_unique()
        # Expecting at least 2 bands (L1/L2) per satellite per epoch for RTK
        expected = n_epochs * n_sats * 2
        actual = self.df.filter(pl.col("obs_type").is_in(["L", "C", "P"])).shape[0]
        return min(100.0, (actual / expected) * 100.0) if expected > 0 else 0.0

    def get_gap_metrics(self):
        """Detects periods with no observations."""
        if len(self.epochs) < 2:
            return {"max_gap": 0, "n_gaps": 0}
        diffs = [
            (self.epochs[i + 1] - self.epochs[i]).total_seconds()
            for i in range(len(self.epochs) - 1)
        ]
        # Assume expected interval is the most common difference
        expected_interval = np.median(diffs)
        gaps = [d for d in diffs if d > expected_interval + 0.1]
        return {
            "max_gap": max(diffs) if diffs else 0,
            "n_gaps": len(gaps),
            "expected_interval": expected_interval,
        }

    def get_integrity_metrics(self):
        """Calculates LLI/Slip rates."""
        slips = self.detect_cycle_slips()
        if self.df.is_empty():
            return {"slip_rate": 0, "total_slips": 0}

        n_sats = self.df["satellite"].n_unique()
        duration_hours = (
            (max(self.epochs) - min(self.epochs)).total_seconds() / 3600.0
            if len(self.epochs) > 1
            else 1.0
        )

        total_slips = len(slips)
        slip_rate_per_sat_hour = (total_slips / n_sats / duration_hours) if n_sats > 0 else 0

        return {"slip_rate": slip_rate_per_sat_hour, "total_slips": total_slips}

    def get_geometric_metrics(self):
        """Assesses geometric distribution (quadrants and elevation spread)."""
        if self.azel_df.is_empty():
            return {"quadrants": 0, "el_spread": 0, "diversity_score": 0}

        # Quadrant coverage (0-90, 90-180, 180-270, 270-360)
        # We take the latest epoch for geometry assessment
        latest_time = self.azel_df["time"].max()
        current_sky = self.azel_df.filter(pl.col("time") == latest_time)

        quads = (
            current_sky.with_columns((pl.col("azimuth") // 90).cast(pl.Int32).alias("quad"))["quad"]
            .unique()
            .to_list()
        )

        n_quads = len(quads)
        el_min = current_sky["elevation"].min()
        el_max = current_sky["elevation"].max()

        # Ensure both are numbers and not None
        if el_min is None or el_max is None:
            el_spread = 0.0
        else:
            # Cast to float to ensure proper numeric operations
            el_min_val = float(el_min) if el_min is not None else 0.0  # type: ignore
            el_max_val = float(el_max) if el_max is not None else 0.0  # type: ignore
            el_spread = abs(el_max_val - el_min_val)

        # Diversity score: 0-100 based on quadrants (60%) and el spread (40%)
        quad_p = (n_quads / 4.0) * 100
        el_p = min(100, (float(el_spread) / 60.0) * 100)  # 60 deg spread is good

        diversity = float(quad_p * 0.6 + el_p * 0.4)

        return {
            "quadrants": n_quads,
            "el_spread": el_spread,
            "diversity_score": diversity,
        }

    def get_per_sat_quality_scores(self):
        """Calculates quality scores for each satellite based on % of 'Good' epochs."""
        # This is now handled within assess_data_quality for efficiency and consistency
        res = self.assess_data_quality()
        return res["sat_scores"]

    def assess_data_quality(self):
        """
        Calculates session quality based on the 4-Step Algorithm in quality.md.
        Strictly epoch-based per-satellite and session evaluation.
        """
        if self.df.is_empty() or self.azel_df.is_empty():
            return {
                "status": "UNCERTAIN",
                "score": 0,
                "reason": "Missing data",
                "sat_scores": pl.DataFrame(),
            }

        # 1. Prepare Data
        snr = self.get_snr()
        mp_est = self.estimate_multipath()
        lli_df = self.df.filter(pl.col("obs_type") == "L")

        primary_bands = ["L1", "G1", "E1", "B1"]
        secondary_bands = ["L2", "G2", "E5b", "B2"]

        obs_l1 = (
            snr.filter(pl.col("frequency").is_in(primary_bands))
            .join(
                mp_est.filter(pl.col("frequency").is_in(primary_bands)).select(
                    ["time", "satellite", "MP"]
                ),
                on=["time", "satellite"],
                how="inner",
            )
            .join(
                lli_df.filter(pl.col("frequency").is_in(primary_bands)).select(
                    ["time", "satellite", "lli"]
                ),
                on=["time", "satellite"],
                how="inner",
            )
            .rename({"value": "snr_l1", "MP": "mp_l1", "lli": "lli_l1"})
        )

        obs_l2 = (
            snr.filter(pl.col("frequency").is_in(secondary_bands))
            .join(
                lli_df.filter(pl.col("frequency").is_in(secondary_bands)).select(
                    ["time", "satellite", "lli"]
                ),
                on=["time", "satellite"],
                how="left",
            )
            .rename({"value": "snr_l2", "lli": "lli_l2"})
            .select(["time", "satellite", "snr_l2", "lli_l2"])
        )

        obs_data = obs_l1.join(obs_l2, on=["time", "satellite"], how="left").join(
            self.azel_df, on=["time", "satellite"], how="inner"
        )

        # 2. Mark "GOOD" satellites (Epoch Level)
        # Criteria: SNR > 35 (L1) / 30 (L2), MP < 1.0, LLI == 0, Elevation > 15
        obs_data = obs_data.with_columns(
            (
                (pl.col("snr_l1") > 35)
                & (pl.col("lli_l1") == 0)
                & (
                    (pl.col("snr_l2").is_null())
                    | ((pl.col("snr_l2") > 30) & (pl.col("lli_l2").fill_null(0) == 0))
                )
                & (pl.col("mp_l1").abs() < 1.0)
                & (pl.col("elevation") > 15)
            ).alias("is_good")
        )

        # 3. Calculate Session Metrics (Epoch by Epoch)
        epoch_stats = []
        for t in self.epochs:
            epoch_obs = obs_data.filter(pl.col("time") == t)
            good_sats = epoch_obs.filter(pl.col("is_good"))
            n_good = good_sats.shape[0]

            if n_good > 0:
                # Step 2: Coverage
                quads = good_sats.with_columns(
                    (pl.col("azimuth") // 90).cast(pl.Int32).alias("quad")
                )
                bins = quads.with_columns(
                    pl.when(pl.col("elevation") < 30)
                    .then(pl.lit("low"))
                    .when(pl.col("elevation") < 50)
                    .then(pl.lit("mid"))
                    .otherwise(pl.lit("high"))
                    .alias("ebin")
                )
                cells_covered = bins.select(["quad", "ebin"]).unique().shape[0]

                # Step 3: Geometry
                el_min = good_sats["elevation"].min()
                el_max = good_sats["elevation"].max()

                if el_min is None or el_max is None:
                    el_span = 0.0
                else:
                    # Cast to float to ensure proper numeric operations
                    el_min_val = float(el_min) if el_min is not None else 0.0  # type: ignore
                    el_max_val = float(el_max) if el_max is not None else 0.0  # type: ignore
                    el_span = abs(el_max_val - el_min_val)

                q_counts = bins.group_by("quad").count()
                if q_counts.shape[0] > 0:
                    min_count = q_counts["count"].min()
                    max_count = q_counts["count"].max()
                    if min_count is not None and max_count is not None and max_count > 0:  # type: ignore
                        # Cast to float to ensure proper numeric operations
                        min_val = float(min_count) if min_count is not None else 0.0  # type: ignore
                        max_val = float(max_count) if max_count is not None else 1.0  # type: ignore
                        balance = min_val / max_val if max_val > 0 else 0.0
                    else:
                        balance = 0.0
                else:
                    balance = 0.0
            else:
                cells_covered = 0
                el_span = 0
                balance = 0

            # Step 4: Final Score per epoch
            s_count = float(min(100.0, (n_good / 20.0) * 100))
            s_cov = float((cells_covered / 12.0) * 100)
            s_el = float(min(100.0, (float(el_span) / 45.0) * 100))
            s_az = float(balance * 100)

            epoch_score = float(s_count * 0.40 + s_cov * 0.30 + s_el * 0.15 + s_az * 0.15)

            epoch_stats.append(
                {
                    "time": t,
                    "n_good": n_good,
                    "cells": cells_covered,
                    "el_span": el_span,
                    "balance": balance,
                    "score": epoch_score,
                }
            )

        epoch_df = pl.DataFrame(epoch_stats)
        session_score = epoch_df["score"].mean()

        # 4. Calculate Per-Satellite Session Scores
        # Defined as the % of epochs where the satellite was marked "GOOD"
        duration_hours = (
            (max(self.epochs) - min(self.epochs)).total_seconds() / 3600.0
            if len(self.epochs) > 1
            else 1.0
        )
        sat_quality = obs_data.group_by("satellite").agg(
            [
                (pl.col("is_good").sum() / pl.col("is_good").count() * 100).alias("total_score"),
                pl.col("snr_l1").mean().alias("snr_l1"),
                pl.col("snr_l2").mean().alias("snr_l2"),
                pl.col("mp_l1").abs().mean().alias("mp_val"),
                # Placeholder for slip counts in the table (LLI is already checked per-epoch)
                (pl.col("lli_l1").sum() + pl.col("lli_l2").fill_null(0).sum()).alias("slip_count"),
            ]
        )

        sat_quality = sat_quality.with_columns(
            pl.when(pl.col("total_score") >= 80)
            .then(pl.lit("Excellent"))
            .when(pl.col("total_score") >= 55)
            .then(pl.lit("Fair"))
            .otherwise(pl.lit("Poor"))
            .alias("rating"),
            (pl.col("slip_count") / duration_hours).alias("slip_rate"),
        ).sort("satellite")

        # 5. Result
        def get_grade(s):
            if s >= 85:
                return "Excellent"
            if s >= 70:
                return "Good"
            if s >= 55:
                return "Fair"
            return "Poor"

        status = get_grade(session_score)

        return {
            "status": status,
            "status_icon": {
                "Excellent": "🟢 EXCELLENT",
                "Good": "🟢 GOOD",
                "Fair": "🟡 FAIR",
                "Poor": "🔴 POOR",
            }.get(status),
            "score": session_score,
            "metrics": {
                "avg_good_sats": epoch_df["n_good"].mean(),
                "avg_cells": epoch_df["cells"].mean(),
                "avg_el_span": epoch_df["el_span"].mean(),
                "avg_balance": epoch_df["balance"].mean(),
            },
            "epoch_df": epoch_df,
            "sat_scores": sat_quality,
            "red_flags": (
                [f"Critical quality drop in {len(epoch_df.filter(pl.col('score') < 55))} epochs"]
                if len(epoch_df.filter(pl.col("score") < 55)) > 0
                else []
            ),
        }

    def get_time_span(self):
        if not self.epochs:
            return None, None
        return min(self.epochs), max(self.epochs)
