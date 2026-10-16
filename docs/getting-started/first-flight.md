# First Flight Tutorial

A complete walkthrough of loading, analyzing, and exporting your first flight data.

## Prerequisites

- PILS installed and working
- A flight directory with data (see [Directory Structure](../user-guide/directory-structure.md))

---

## Step 1: Discover Available Flights

```python
from pils.loader.path import PathLoader

# Initialize loader
loader = PathLoader(base_path="/data/campaigns")

# List all flights
all_flights = loader.load_all_flights()
print(f"Found {len(all_flights)} flights")

# List flights for a specific date
dec_flights = loader.load_flights_by_date(date="2025-12-08")
for f in dec_flights:
    print(f"  - {f['flight_name']}")
```

**Output:**
```
Found 15 flights
  - flight_20251208_1506
  - flight_20251208_1623
  - flight_20251208_1745
```

---

## Step 2: Load Flight Metadata

```python
# Load single flight
flight_info = loader.load_single_flight(
    flight_id="flight_20251208_1506"
)

# Inspect metadata
print("Flight Info:")
for key, value in flight_info.items():
    print(f"  {key}: {value}")
```

**Output:**
```
Flight Info:
  flight_name: flight_20251208_1506
  flight_path: /data/campaigns/202511/20251208/flight_20251208_1506
  date: 2025-12-08
  campaign: 202511
  aux_path: /data/campaigns/202511/20251208/flight_20251208_1506/aux
  sensors_path: /data/campaigns/202511/20251208/flight_20251208_1506/aux/sensors
  drone_data_path: /data/campaigns/202511/20251208/flight_20251208_1506/aux/drone_data
```

---

## Step 3: Create Flight Container

```python
from pils.flight import Flight

# Create container
flight = Flight(flight_info)

# Access basic properties
print(f"Flight Name: {flight.flight_name}")
print(f"Flight Date: {flight.date}")
print(f"Campaign: {flight.campaign}")
```

---

## Step 4: Load Drone Data

```python
# Auto-detect and load drone platform
flight.add_drone_data()

# Check what was loaded
print(f"Drone data keys: {list(flight.drone_data.keys())}")
```

=== "DJI Output"

    ```
    Drone data keys: ['data']
    ```

=== "BlackSquare Output"

    ```
    Drone data keys: ['GPS', 'IMU', 'BARO', 'ATT', 'MAG', 'BAT', 'RCOU', 'POS']
    ```

### Inspect Drone Data

```python
import polars as pl

# For DJI
if 'data' in flight.drone_data:
    df = flight.drone_data['data']
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns[:5]}...")
    print(df.head(3))
```

**Output:**
```
Shape: (5420, 18)
Columns: ['tick', 'datetime', 'latitude', 'longitude', 'altitude']...
┌──────┬─────────────────────┬──────────┬───────────┬──────────┐
│ tick │ datetime            │ latitude │ longitude │ altitude │
│ ---  │ ---                 │ ---      │ ---       │ ---      │
│ i64  │ datetime[μs]        │ f64      │ f64       │ f64      │
╞══════╪═════════════════════╪══════════╪═══════════╪══════════╡
│ 0    │ 2025-12-08 15:06:23 │ 40.7128  │ -74.006   │ 10.5     │
│ 1    │ 2025-12-08 15:06:24 │ 40.7129  │ -74.0061  │ 10.8     │
│ 2    │ 2025-12-08 15:06:25 │ 40.713   │ -74.0062  │ 11.2     │
└──────┴─────────────────────┴──────────┴───────────┴──────────┘
```

---

## Step 5: Load Sensor Data

```python
# Load specific sensors
flight.add_sensor_data(['gps', 'imu'])

# Check loaded sensors
print(f"Loaded sensors: {list(flight.raw_data.keys())}")
```

### Access GPS Data

```python
# Payload GPS table (column names: see the Data Schema page)
gps_df = flight.raw_data.payload_data.gps
print(f"GPS points: {gps_df.height}")
print(f"Columns: {gps_df.columns}")
```

**Output:**
```
GPS points: 1687
Columns: ['timestamp', 'unix_time_ms', 'ecef_x_m', 'ecef_y_m', 'ecef_z_m', ...,
          'lat_deg', 'lon_deg', 'h_ell_m', 'h_msl_m', 'h_acc_m', 'v_acc_m', ...]
```

### Access IMU Data

```python
# One table per sub-sensor that has data
imu = flight.raw_data.payload_data.imu

print(f"Accelerometer samples: {imu['accelerometer'].height}")
print(f"Gyroscope samples: {imu['gyroscope'].height}")
print(f"Magnetometer samples: {imu['magnetometer'].height}")
print(f"Barometer samples: {imu['barometer'].height}")
```

**Output:**
```
Accelerometer samples: 131848
Gyroscope samples: 131869
Magnetometer samples: 131871
Barometer samples: 131848
```

---

## Step 6: Time Synchronization

### Using Flight.sync() (Recommended)

```python
# Synchronize all loaded sources on the payload GPS time base
synced = flight.sync(target_rate={"drone": 10.0, "payload": 10.0})

# One table per source, all on the same timestamps
print({name: df.shape for name, df in synced.items()})
print(synced["drone"].select("timestamp", "lat_deg", "lon_deg", "h_ell_m").head())
print(flight.sync_offsets)  # seconds added to each source's clock
```

**Output:**
```
{'drone': (8431, 25), 'litchi': (8431, 19), 'inclinometer': (8431, 20), 'adc': (8431, 2), ...}
┌──────────────┬────────────┬────────────┬──────────┐
│ timestamp    ┆ lat_deg    ┆ lon_deg    ┆ h_ell_m  │
│ ---          ┆ ---        ┆ ---        ┆ ---      │
│ f64          ┆ f64        ┆ f64        ┆ f64      │
╞══════════════╪════════════╪════════════╪══════════╡
│ 1.7652e9     ┆ null       ┆ null       ┆ null     │
│ ...          ┆ ...        ┆ ...        ┆ ...      │
└──────────────┴────────────┴────────────┴──────────┘
{'drone_gps': 0.196, 'litchi_gps': -0.187, 'inclinometer': 0.306}
```

Values are null where a source has no data (before take-off, after landing).
See [Time Synchronization](../user-guide/synchronization.md) for the rules.

### Manual Synchronization (Advanced)

```python
from pils.synchronizer import Synchronizer

payload = flight.raw_data.payload_data
sync = Synchronizer()

# Reference GPS (from the sensor payload)
sync.add_gps_reference(payload.gps)

# Drone RTK positions
sync.add_drone_gps(
    flight.raw_data.drone_data.drone,
    lat_col="rtk_lat_deg",
    lon_col="rtk_lon_deg",
    alt_col="rtk_h_ell_m",
)

# Other sensors
sync.add_payload_sensor("adc", payload.adc)

result = sync.synchronize(target_rate={"drone": 10.0, "payload": 10.0})
print(sync.get_offset_summary())
```

---

## Step 7: Export Data

### Save the Flight

`Flight.save` writes the raw data and, if the flight is synchronized, a new
synchronized revision, as Parquet tables with a manifest (see
[Directory Structure](../user-guide/directory-structure.md)).
`pils package` saves to `<flight>/proc/flight/`.

```python
from pathlib import Path

output_dir = Path("./output")

# raw/ and sync/rev_YYYYMMDD_hhmmss/
revision = flight.save(output_dir / "flight")
print(f"Saved synchronized revision {revision}")
```

### Export Individual DataFrames

```python
# To CSV
gps_df.write_csv(output_dir / "gps.csv")

# To Parquet (efficient binary)
gps_df.write_parquet(output_dir / "gps.parquet")

# To JSON
gps_df.write_json(output_dir / "gps.json")
```

### Reload

```python
# Later, reload the flight (latest synchronized revision)
reloaded = Flight.load(output_dir / "flight")
print(list(reloaded.sync_data))
```

Flights saved in HDF5 by earlier versions still load with
`Flight.from_hdf5(path)`; their columns are renamed to the schema.

---

## Complete Script

```python
"""First Flight Analysis - Complete Example"""
from pathlib import Path
import polars as pl

from pils.loader.path import PathLoader
from pils.flight import Flight

# Configuration
CAMPAIGNS = Path("/data/campaigns")
FLIGHT_ID = "flight_20251208_1506"
OUTPUT = Path("./output")
OUTPUT.mkdir(exist_ok=True)

# Load flight
print("Loading flight...")
loader = PathLoader(base_path=CAMPAIGNS)
flight_info = loader.load_single_flight(flight_id=FLIGHT_ID)

# Create container
flight = Flight(flight_info)
flight.add_drone_data()
flight.add_sensor_data(['gps', 'imu', 'adc'])

# Analyze GPS
print("\nGPS Analysis:")
gps_df = flight.raw_data.payload_data.gps
accurate = gps_df.filter(pl.col('h_acc_m') < 2.0)
print(f"  Total points: {gps_df.height}")
print(f"  Accuracy < 2 m: {accurate.height} ({accurate.height/gps_df.height:.1%})")

# Analyze IMU
print("\nIMU Analysis:")
imu = flight.raw_data.payload_data.imu
accel = imu['accelerometer']
print(f"  Samples: {accel.height}")
print(f"  Duration: {(accel['timestamp'].max() - accel['timestamp'].min()):.1f} seconds")

# Synchronize all data
print("\nSynchronization:")
synced = flight.sync(target_rate={"drone": 10.0, "payload": 10.0})
print(f"  Synchronized tables: {list(synced)}")
print(f"  Drone: {synced['drone'].shape}")

# Save
print("\nSaving...")
revision = flight.save(OUTPUT / FLIGHT_ID)  # raw/ and sync/<revision>/
gps_df.write_csv(OUTPUT / "gps.csv")
print(f"  Saved revision {revision} to {OUTPUT / FLIGHT_ID}")

print("\nDone!")
```

---

## Next Steps

- [User Guide](../user-guide/index.md) - Detailed feature documentation
- [PPK Analysis](../api/analysis/ppk.md) - Post-processed kinematics
- [API Reference](../api/index.md) - Complete API documentation
