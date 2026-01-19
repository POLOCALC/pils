#!/usr/bin/env python3
"""Compare DAT decoding with DATCON CSV output to verify tick extraction."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from pils.drones.DJIDrone import DJIDrone


def main():
    print("ok")
    base_path = Path(
        "/mnt/data/POLOCALC/campaigns/202511/20251201/flight_20251201_1515/drone"
    )

    dat_file = base_path / "20251201_151517_drone.dat"
    csv_file = "data_1515.csv"

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load DAT file
    print("\n1. Loading from DAT file...")
    drone_dat = DJIDrone(dat_file)
    drone_dat.load_data(use_dat=True)

    gps_data_df = drone_dat.data["GPS"]

    with pl.Config(tbl_rows=20):
        print(gps_data_df["datetime"])

    # Load CSV file
    print("\n2. Loading from CSV file (DATCON)...")
    drone_csv = DJIDrone(csv_file)
    drone_csv.load_data(use_dat=False, cols=None)  # Load all columns
    csv_data = drone_csv.data["CSV"]

    fig = plt.figure(figsize=(20, 10))

    # Plot 1: GPS Longitude vs Datetime - Compare DAT and CSV
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(gps_data_df["tick"], gps_data_df["GPS:longitude"], label="DAT")
    ax1.plot(csv_data["Clock:Tick#"], csv_data["GPS:Long[degrees]"], label="CSV")
    ax1.set_title("GPS Longitude Comparison")
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(gps_data_df["tick"], gps_data_df["timestamp"], label="DAT")
    ax2.plot(csv_data["Clock:Tick#"], csv_data["timestamp"], label="CSV")
    ax2.set_title("GPS Longitude Comparison")
    ax2.set_ylim(csv_data["timestamp"].min(), csv_data["timestamp"].max())
    ax2.legend()

    plt.savefig("compare_dat_csv.png")

    aligned = drone_dat.align_datfile()

    print("\nAligned Dataframe shape:", aligned.shape)

    print("\nCorrected Tick and Correct Timestamp (High Precision):")

    print(aligned.select(["corrected_tick", "correct_timestamp", "datetime_converted"]))

    # print("\nCorrect Timestamp to Datetime conversion (with ms precision):")
    # # Convert timestamp (seconds) to datetime with ms precision
    # temp_df = aligned.select(
    #     [
    #         pl.col("correct_timestamp"),
    #         (pl.col("correct_timestamp") * 1000)
    #         .cast(pl.Int64)
    #         .cast(pl.Datetime("ms"))
    #         .alias("datetime_converted"),
    #     ]
    # )

    # print(temp_df)


if __name__ == "__main__":
    main()
