#!/usr/bin/env python3
"""
Compare GPS data from UBX binary file with CSV telemetry data using STOUT database interface.

This script loads flight information from the STOUT database, retrieves GPS and drone telemetry,
and compares GPS coordinates, timestamps, and quality metrics.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path("/home/stout")))

from pils.drones.DJIDrone import DJIDrone
from pils.sensors.gps import GPS
from stout.database.manager import DatabaseManager
from stout.services.campaigns.service import CampaignService


def load_flight_data(flight_date_time=None, flight_uuid=None):
    """
    Load flight data from STOUT database.

    Parameters
    ----------
    flight_date_time : tuple, optional
        Tuple of (year, month, day, hour, minute) to find flight
    flight_uuid : str, optional
        Flight UUID in the database

    Returns
    -------
    dict
        Dictionary with flight information and all paths
    """
    from datetime import datetime

    db_manager = DatabaseManager()
    campaign_service = CampaignService(db_manager)

    flight_info = None

    if flight_uuid:
        # Search by UUID
        flight_info = campaign_service.get_flight(flight_id=flight_uuid)
    elif flight_date_time:
        # Search by date/time using new method
        target_date = datetime(*flight_date_time[:3])
        target_hour = flight_date_time[3] if len(flight_date_time) > 3 else None
        target_min = flight_date_time[4] if len(flight_date_time) > 4 else None

        flight_info = campaign_service.get_flight_by_datetime(
            date=target_date, hour=target_hour, minute=target_min
        )

    if not flight_info:
        search_hint = (
            f"UUID: {flight_uuid}" if flight_uuid else f"Date/Time: {flight_date_time}"
        )
        raise ValueError(f"Flight not found in database ({search_hint})")

    return flight_info


def get_gps_and_drone_paths(flight_info: dict) -> tuple[Path, Path]:
    """
    Extract GPS and drone data file paths from flight info.

    Uses pils library to locate data files based on flight folder.

    Parameters
    ----------
    flight_info : dict
        Flight information dictionary from database

    Returns
    -------
    tuple[Path, Path]
        GPS file path, Drone telemetry file path
    """
    flight = flight_info

    # Get base paths from database
    aux_path = flight.get("aux_data_folder_path")
    drone_path_base = flight.get("drone_data_folder_path")
    flight_name = flight.get("flight_name")

    if not aux_path or not drone_path_base:
        raise ValueError("Flight paths not found in database")

    # Construct file paths
    aux_path = Path(aux_path)
    drone_path_base = Path(drone_path_base)

    # GPS file - usually in aux/sensors folder
    gps_candidates = list(aux_path.glob("*GPS*.ubx")) + list(
        aux_path.glob("sensors/*GPS*.ubx")
    )
    if not gps_candidates:
        gps_candidates = list(aux_path.glob("*GPS*.bin")) + list(
            aux_path.glob("sensors/*GPS*.bin")
        )

    if gps_candidates:
        gps_path = gps_candidates[0]
    else:
        raise FileNotFoundError(f"GPS file not found in {aux_path}")

    # Drone telemetry file
    drone_candidates = list(drone_path_base.glob("*drone.dat")) + list(
        drone_path_base.glob(f"*{flight_name}*drone.dat")
    )

    if drone_candidates:
        drone_path = drone_candidates[0]
    else:
        raise FileNotFoundError(f"Drone telemetry file not found in {drone_path_base}")

    return gps_path, drone_path


def load_gps_data(gps_path: Path) -> pl.DataFrame:
    """
    Load and process GPS data from UBX binary file.

    Parameters
    ----------
    gps_path : Path
        Path to GPS UBX binary file

    Returns
    -------
    pl.DataFrame
        GPS data with all NAV message types merged
    """
    gps = GPS(str(gps_path))
    gps.load_data()
    if gps.data is None:
        raise ValueError(f"Failed to load GPS data from {gps_path}")
    return gps.data


def load_drone_telemetry(drone_path: Path) -> pl.DataFrame:
    """
    Load drone telemetry from DAT file.

    Parameters
    ----------
    drone_path : Path
        Path to drone DAT file

    Returns
    -------
    pl.DataFrame
        Drone telemetry data including GPS coordinates
    """
    drone = DJIDrone(str(drone_path))
    drone.load_data(use_dat=True)

    if "GPS" in drone.data:
        return drone.data["GPS"]
    else:
        raise ValueError("No GPS data found in drone telemetry")


def compare_gps_coordinates(gps_df: pl.DataFrame, drone_df: pl.DataFrame) -> dict:
    """
    Compare GPS coordinates between UBX and DAT sources.

    Parameters
    ----------
    gps_df : pl.DataFrame
        GPS data from UBX file
    drone_df : pl.DataFrame
        Drone telemetry data from DAT file

    Returns
    -------
    dict
        Comparison statistics (differences, correlations, etc.)
    """
    comparison = {}

    # Extract common fields
    gps_cols = gps_df.columns

    # Get latitude/longitude from GPS data (from POSLLH message)
    if "posllh_lat" in gps_cols and "posllh_lon" in gps_cols:
        gps_lats = gps_df.select("posllh_lat").to_numpy().flatten()
        gps_lons = gps_df.select("posllh_lon").to_numpy().flatten()
    else:
        print("Warning: POSLLH coordinates not found in GPS data")
        gps_lats = gps_lons = None

    # Get drone coordinates
    if "GPS:latitude" in drone_df.columns and "GPS:longitude" in drone_df.columns:
        drone_lats = drone_df.select("GPS:latitude").to_numpy().flatten()
        drone_lons = drone_df.select("GPS:longitude").to_numpy().flatten()
    else:
        print("Warning: GPS coordinates not found in drone data")
        drone_lats = drone_lons = None

    if (
        gps_lats is not None
        and drone_lats is not None
        and gps_lons is not None
        and drone_lons is not None
    ):
        # Align data by length (take minimum)
        min_len = min(len(gps_lats), len(drone_lats))
        gps_lats = gps_lats[:min_len]
        gps_lons = gps_lons[:min_len]
        drone_lats = drone_lats[:min_len]
        drone_lons = drone_lons[:min_len]

        # Calculate differences
        lat_diff = gps_lats - drone_lats
        lon_diff = gps_lons - drone_lons

        # Calculate distance difference (simple Euclidean in degrees)
        dist_diff = np.sqrt(lat_diff**2 + lon_diff**2)

        comparison["latitude"] = {
            "mean_diff": float(np.mean(lat_diff)),
            "std_diff": float(np.std(lat_diff)),
            "max_diff": float(np.max(np.abs(lat_diff))),
        }
        comparison["longitude"] = {
            "mean_diff": float(np.mean(lon_diff)),
            "std_diff": float(np.std(lon_diff)),
            "max_diff": float(np.max(np.abs(lon_diff))),
        }
        comparison["distance"] = {
            "mean_diff": float(np.mean(dist_diff)),
            "std_diff": float(np.std(dist_diff)),
            "max_diff": float(np.max(dist_diff)),
        }

    return comparison


def compare_timestamps(gps_df: pl.DataFrame, drone_df: pl.DataFrame) -> dict:
    """
    Compare timestamps between UBX and DAT sources.

    Parameters
    ----------
    gps_df : pl.DataFrame
        GPS data from UBX file
    drone_df : pl.DataFrame
        Drone telemetry data from DAT file

    Returns
    -------
    dict
        Timestamp comparison statistics
    """
    comparison = {}

    # Get timestamps
    if "timestamp" in gps_df.columns and "timestamp" in drone_df.columns:
        gps_ts = gps_df.select("timestamp").to_numpy().flatten()
        drone_ts = drone_df.select("timestamp").to_numpy().flatten()

        # Align data
        min_len = min(len(gps_ts), len(drone_ts))
        gps_ts = gps_ts[:min_len]
        drone_ts = drone_ts[:min_len]

        # Calculate differences
        ts_diff = gps_ts - drone_ts

        comparison["timestamp"] = {
            "mean_diff": float(np.mean(ts_diff)),
            "std_diff": float(np.std(ts_diff)),
            "max_diff": float(np.max(np.abs(ts_diff))),
            "num_samples": int(min_len),
        }

    return comparison


def plot_comparison(gps_df: pl.DataFrame, drone_df: pl.DataFrame, output_path: str):
    """
    Create comparison plots between GPS and drone telemetry.

    Parameters
    ----------
    gps_df : pl.DataFrame
        GPS data from UBX file
    drone_df : pl.DataFrame
        Drone telemetry data from DAT file
    output_path : str
        Path to save output plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Extract data
    gps_cols = gps_df.columns
    drone_cols = drone_df.columns

    # Plot 1: Latitude comparison
    if "posllh_lat" in gps_cols and "GPS:latitude" in drone_cols:
        ax = axes[0, 0]
        gps_lats = gps_df.select("posllh_lat").to_numpy().flatten()
        drone_lats = drone_df.select("GPS:latitude").to_numpy().flatten()
        min_len = min(len(gps_lats), len(drone_lats))
        ax.plot(gps_lats[:min_len], label="UBX GPS", alpha=0.7)
        ax.plot(drone_lats[:min_len], label="DAT Telemetry", alpha=0.7)
        ax.set_title("Latitude Comparison")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Latitude [degrees]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 2: Longitude comparison
    if "posllh_lon" in gps_cols and "GPS:longitude" in drone_cols:
        ax = axes[0, 1]
        gps_lons = gps_df.select("posllh_lon").to_numpy().flatten()
        drone_lons = drone_df.select("GPS:longitude").to_numpy().flatten()
        min_len = min(len(gps_lons), len(drone_lons))
        ax.plot(gps_lons[:min_len], label="UBX GPS", alpha=0.7)
        ax.plot(drone_lons[:min_len], label="DAT Telemetry", alpha=0.7)
        ax.set_title("Longitude Comparison")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Longitude [degrees]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 3: Timestamp comparison
    if "timestamp" in gps_df.columns and "timestamp" in drone_df.columns:
        ax = axes[0, 2]
        gps_ts = gps_df.select("timestamp").to_numpy().flatten()
        drone_ts = drone_df.select("timestamp").to_numpy().flatten()
        min_len = min(len(gps_ts), len(drone_ts))
        ax.plot(gps_ts[:min_len], label="UBX GPS", alpha=0.7)
        ax.plot(drone_ts[:min_len], label="DAT Telemetry", alpha=0.7)
        ax.set_title("Timestamp Comparison")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Timestamp [s]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 4: Geographic scatter
    if (
        "posllh_lat" in gps_cols
        and "posllh_lon" in gps_cols
        and "GPS:latitude" in drone_cols
        and "GPS:longitude" in drone_cols
    ):
        ax = axes[1, 0]
        gps_lats = gps_df.select("posllh_lat").to_numpy().flatten()
        gps_lons = gps_df.select("posllh_lon").to_numpy().flatten()
        drone_lats = drone_df.select("GPS:latitude").to_numpy().flatten()
        drone_lons = drone_df.select("GPS:longitude").to_numpy().flatten()
        min_len = min(len(gps_lats), len(drone_lats))
        ax.scatter(
            gps_lons[:min_len], gps_lats[:min_len], label="UBX GPS", alpha=0.5, s=5
        )
        ax.scatter(
            drone_lons[:min_len],
            drone_lats[:min_len],
            label="DAT Telemetry",
            alpha=0.5,
            s=5,
        )
        ax.set_title("Geographic Position")
        ax.set_xlabel("Longitude [degrees]")
        ax.set_ylabel("Latitude [degrees]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 5: Latitude vs Timestamp scatter
    if (
        "posllh_lat" in gps_cols
        and "timestamp" in gps_df.columns
        and "GPS:latitude" in drone_cols
        and "timestamp" in drone_df.columns
    ):
        ax = axes[1, 1]
        gps_lats = gps_df.select("posllh_lat").to_numpy().flatten()
        gps_ts = gps_df.select("timestamp").to_numpy().flatten()
        drone_lats = drone_df.select("GPS:latitude").to_numpy().flatten()
        drone_ts = drone_df.select("timestamp").to_numpy().flatten()
        min_len = min(len(gps_lats), len(drone_lats))
        ax.scatter(
            gps_ts[:min_len], gps_lats[:min_len], label="UBX GPS", alpha=0.5, s=3
        )
        ax.scatter(
            drone_ts[:min_len],
            drone_lats[:min_len],
            label="DAT Telemetry",
            alpha=0.5,
            s=3,
        )
        ax.set_title("Latitude vs Timestamp")
        ax.set_xlabel("Timestamp [s]")
        ax.set_ylabel("Latitude [degrees]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot 6: Longitude vs Timestamp scatter
    if (
        "posllh_lon" in gps_cols
        and "timestamp" in gps_df.columns
        and "GPS:longitude" in drone_cols
        and "timestamp" in drone_df.columns
    ):
        ax = axes[1, 2]
        gps_lons = gps_df.select("posllh_lon").to_numpy().flatten()
        gps_ts = gps_df.select("timestamp").to_numpy().flatten()
        drone_lons = drone_df.select("GPS:longitude").to_numpy().flatten()
        drone_ts = drone_df.select("timestamp").to_numpy().flatten()
        min_len = min(len(gps_lons), len(drone_lons))
        ax.scatter(
            gps_ts[:min_len], gps_lons[:min_len], label="UBX GPS", alpha=0.5, s=3
        )
        ax.scatter(
            drone_ts[:min_len],
            drone_lons[:min_len],
            label="DAT Telemetry",
            alpha=0.5,
            s=3,
        )
        ax.set_title("Longitude vs Timestamp")
        ax.set_xlabel("Timestamp [s]")
        ax.set_ylabel("Longitude [degrees]")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Comparison plot saved to {output_path}")
    plt.close()


def main():
    """Main comparison workflow."""

    # Flight parameters - using UUID from database
    flight_uuid = "e809e9fe-0dcb-4a70-8300-ac9dceb99d3f"  # Flight from 2025-12-01 15:15
    # Alternative: use date/time tuple
    # flight_date_time = (2025, 12, 1, 15, 15)

    print("=" * 80)
    print("GPS vs DRONE TELEMETRY COMPARISON")
    print("=" * 80)

    try:
        # Load flight info from database
        print(f"\n1. Loading flight from STOUT database...")
        flight_info = load_flight_data(flight_uuid=flight_uuid)

        print(f"   Flight Name: {flight_info.get('flight_name')}")
        print(f"   Takeoff: {flight_info.get('takeoff_datetime')}")
        print(f"   Landing: {flight_info.get('landing_datetime')}")
        print(f"   Aux Path: {flight_info.get('aux_data_folder_path')}")
        print(f"   Drone Path: {flight_info.get('drone_data_folder_path')}")

        # Get file paths
        print("\n2. Locating GPS and drone telemetry files...")
        gps_path, drone_path = get_gps_and_drone_paths(flight_info)

        # Load GPS data
        print("\n3. Loading GPS data from UBX binary...")
        gps_df = load_gps_data(gps_path)
        print(f"   GPS data shape: {gps_df.shape}")
        print(f"   GPS columns: {gps_df.columns[:10]}...")

        # Load drone telemetry
        print("\n4. Loading drone telemetry from DAT file...")
        drone_df = load_drone_telemetry(drone_path)
        print(f"   Drone data shape: {drone_df.shape}")
        print(f"   Drone columns: {drone_df.columns[:10]}...")

        # Compare
        print("\n5. Comparing GPS coordinates...")
        coord_comparison = compare_gps_coordinates(gps_df, drone_df)
        for key, val in coord_comparison.items():
            print(f"\n   {key.upper()}:")
            for metric, value in val.items():
                print(f"      {metric}: {value}")

        print("\n6. Comparing timestamps...")
        ts_comparison = compare_timestamps(gps_df, drone_df)
        if ts_comparison:
            for key, val in ts_comparison.items():
                print(f"\n   {key.upper()}:")
                for metric, value in val.items():
                    print(f"      {metric}: {value}")

        # Generate plots
        print("\n7. Generating comparison plots...")
        plot_comparison(gps_df, drone_df, "gps_vs_drone_comparison.png")

        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
