# run_single_day.py
import sys
print("🔧 Imports starting...", flush=True)
from pils.loader.path import PathLoader
from pils.flight import Flight
from pils.sensors.camera import Camera, PhotogrammetryConfig
print("✅ Imports done", flush=True)

# ------------------------------------------------------------------
# Input day
# ------------------------------------------------------------------
day = sys.argv[1]
campaign_name = "202511"
print(f"Processing day: {day}", flush=True)

# ------------------------------------------------------------------
# Load flights ONLY for this day
# ------------------------------------------------------------------
print("📂 Loading all campaign flights...", flush=True)
loader = PathLoader("/data/POLOCALC/")
all_flights_meta = loader.load_all_campaign_flights(
    campaign_name=campaign_name
)
print(f"📂 Total flights in campaign: {len(all_flights_meta)}", flush=True)

flights_meta = [
    f for f in all_flights_meta
    if day in str(f)
]
print(f"📂 Flights for day {day}: {len(flights_meta)}", flush=True)

if not flights_meta:
    raise RuntimeError(f"No flights found for {day}")

for i, f in enumerate(flights_meta):
    print(f"   [{i}] {f}", flush=True)

# ------------------------------------------------------------------
# Detect camera model
# ------------------------------------------------------------------
print("\n📷 Detecting camera model from first flight...", flush=True)
print(f"   Flight path: {flights_meta[0]}", flush=True)

first_flight = Flight(flights_meta[0])
print("   Flight object created", flush=True)

print("   Reading log time...", flush=True)
first_flight.add_camera_data(use_photogrammetry=False)
print("   add_camera_data done", flush=True)

camera_obj = first_flight.raw_data.payload_data.camera_obj
print(f"   camera_obj: {camera_obj}", flush=True)
print(f"   is_image_sequence: {getattr(camera_obj, 'is_image_sequence', False)}", flush=True)
print(f"   fps: {getattr(camera_obj, 'fps', None)}", flush=True)

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
cfg = PhotogrammetryConfig(
    "/home/fastori/Desktop/ARS/pils/pils/config/photogrammetryConfig.yaml",
    camera_model=camera_model,
)
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
output_dir = f"/home/fastori/Desktop/ARS/photogrammetry_results_1225/{day}"
csv_file   = f"/data/POLOCALC/campaigns/{campaign_name}/metadata/{campaign_name}_coordinates.csv"

print(f"\n🚀 Starting photogrammetry pipeline", flush=True)
print(f"   Output dir : {output_dir}", flush=True)
print(f"   CSV file   : {csv_file}", flush=True)
print(f"   Flights    : {len(all_flights)}", flush=True)

result = Camera.run_photogrammetry_multi_flights(
    flights=all_flights,
    csv_file=csv_file,
    config=cfg,
    output_dir=output_dir,
    start_from_dict=None,
)

print(f"\n✅ Done: {result.shape[0]} frames × {result.shape[1]} columns", flush=True)