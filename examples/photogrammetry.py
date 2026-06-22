# run_single_day.py
import sys
import argparse
from pathlib import Path

print("🔧 Imports starting...", flush=True)
from pils.loader.path import PathLoader
from pils.flight import Flight
from pils.sensors.camera import Camera, PhotogrammetryConfig
print("✅ Imports done", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run photogrammetry pipeline for a single campaign day.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--day",            required=True,  help="Day string to filter flights (e.g. 20251201).")
    parser.add_argument("--campaign-name",  default="202511", help="Campaign name.")
    parser.add_argument("--data-root",      default="/data/POLOCALC/", help="Root data directory.")
    parser.add_argument(
        "--config",
        default="/home/fastori/Desktop/ARS/pils/pils/config/photogrammetryConfig.yaml",
        help="Path to photogrammetry YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to /home/fastori/Desktop/ARS/photogrammetry_results_1225/<day>/",
    )
    parser.add_argument(
        "--csv-file",
        default=None,
        help="Geodetic targets CSV. Defaults to <data-root>/campaigns/<campaign-name>/metadata/<campaign-name>_coordinates.csv",
    )
    parser.add_argument(
        "--start_from_dict",
        default=None,
        help="Resume from an intermediate .ecsv file (skips step p2).",
    )
    parser.add_argument(
        "--target-indices",
        nargs="+",
        type=int,
        default=None,
        metavar="I",
        help="Target indices to use (e.g. --target-indices 0 2 4). Uses all if omitted.",
    )

    parser.add_argument(
        "--mcmc_solution",
        type=bool,
        default=False,
        help="Whether to use MCMC solution for photogrammetry (instead of least squares)."

    )

    return parser.parse_args()


def main():
    args = parse_args()

    day           = args.day
    campaign_name = args.campaign_name
    data_root     = args.data_root
    output_dir    = args.output_dir or f"/home/fastori/Desktop/ARS/photogrammetry_results_1225/{day}"
    csv_file      = args.csv_file   or f"{data_root}/campaigns/{campaign_name}/metadata/{campaign_name}_coordinates.csv"

    print(f"Processing day: {day}", flush=True)

    # ------------------------------------------------------------------
    # Load flights for this day
    # ------------------------------------------------------------------
    print("📂 Loading all campaign flights...", flush=True)
    loader = PathLoader(data_root)
    all_flights_meta = loader.load_all_campaign_flights(campaign_name=campaign_name)
    print(f"📂 Total flights in campaign: {len(all_flights_meta)}", flush=True)

    flights_meta = [f for f in all_flights_meta if day in str(f)]
    print(f"📂 Flights for day {day}: {len(flights_meta)}", flush=True)

    if not flights_meta:
        raise RuntimeError(f"No flights found for {day}")

    for i, f in enumerate(flights_meta):
        print(f"   [{i}] {f}", flush=True)

    # ------------------------------------------------------------------
    # Detect camera model from first flight
    # ------------------------------------------------------------------
    print("\n📷 Detecting camera model from first flight...", flush=True)
    print(f"   Flight path: {flights_meta[0]}", flush=True)
    first_flight = Flight(flights_meta[0])
    first_flight.add_camera_data(use_photogrammetry=False)
    camera_obj = first_flight.raw_data.payload_data.camera_obj
    print(f"   camera_obj: {camera_obj}", flush=True)

    camera_model = (
        "alvium"
        if getattr(camera_obj, "is_image_sequence", False)
        or getattr(camera_obj, "fps", None) == 1.0
        else "sony"
    )
    print(f"✅ Selected camera model: {camera_model}", flush=True)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    print("\n⚙️  Loading PhotogrammetryConfig...", flush=True)
    cfg = PhotogrammetryConfig(args.config, camera_model=camera_model)
    print(f"✅ Config loaded: {cfg}", flush=True)

    # ------------------------------------------------------------------
    # Build Flight objects
    # ------------------------------------------------------------------
    print("\n🛩️  Building all Flight objects...", flush=True)
    all_flights = [first_flight] + [Flight(meta) for meta in flights_meta[1:]]
    print(f"✅ Built {len(all_flights)} Flight objects", flush=True)

    # ------------------------------------------------------------------
    # Run photogrammetry
    # ------------------------------------------------------------------
    print(f"\n🚀 Starting photogrammetry pipeline", flush=True)
    print(f"   Output dir    : {output_dir}", flush=True)
    print(f"   CSV file      : {csv_file}", flush=True)
    print(f"   Flights       : {len(all_flights)}", flush=True)
    print(f"   Target indices: {args.target_indices or 'all'}", flush=True)

    result = Camera.run_photogrammetry_multi_flights(
        flights=all_flights,
        csv_file=csv_file,
        config=cfg,
        output_dir=output_dir,
        start_from_dict=args.start_from_dict,
        target_indices=args.target_indices,
        mcmc_solution=args.mcmc_solution,
    )

    print(f"\n✅ Done: {result.shape[0]} frames × {result.shape[1]} columns", flush=True)


if __name__ == "__main__":
    main()