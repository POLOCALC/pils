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
# Get GPS sensor object
gps = flight['gps']

# Access data
gps_df = gps.data
print(f"GPS points: {gps_df.height}")
print(f"Columns: {gps_df.columns}")
```

**Output:**
```
GPS points: 3500
Columns: ['timestamp', 'latitude', 'longitude', 'altitude', 'fix_quality', 
          'num_satellites', 'hdop', 'velocity_north', 'velocity_east']
```

### Access IMU Data

```python
# Get IMU sensor object
imu = flight['imu']

# IMU has multiple sub-sensors
print(f"Accelerometer samples: {imu.accelerometer.height}")
print(f"Gyroscope samples: {imu.gyroscope.height}")
print(f"Magnetometer samples: {imu.magnetometer.height}")
print(f"Barometer samples: {imu.barometer.height}")
```

**Output:**
```
Accelerometer samples: 35000
Gyroscope samples: 35000
Magnetometer samples: 35000
Barometer samples: 17500
```

---

## Step 6: Time Synchronization

### Using Flight.sync() (Recommended)

```python
# Synchronize all loaded sensors to a common time base
synced_df = flight.sync(target_rate_hz=10.0)

print(f"Synchronized data: {synced_df.shape}")
print(f"Columns: {synced_df.columns}")
print(f"Time range: {synced_df['timestamp'].min():.1f}s - {synced_df['timestamp'].max():.1f}s")

# Synchronized data is stored in flight.sync_data
print(f"\nFirst few rows:")
print(synced_df.head())
```

**Output:**
```
Synchronized data: (3500, 25)
Columns: ['timestamp', 'gps_latitude', 'gps_longitude', 'gps_altitude', 
          'drone_latitude', 'drone_longitude', 'accel_ax', 'accel_ay', 'accel_az', ...]
Time range: 0.0s - 350.0s

First few rows:
┌───────────┬──────────────┬───────────────┬──────────────┬────────┐
│ timestamp │ gps_latitude │ gps_longitude │ gps_altitude │ ...    │
│ ---       │ ---          │ ---           │ ---          │        │
│ f64       │ f64          │ f64           │ f64          │        │
╞═══════════╪══════════════╪═══════════════╪══════════════╪════════╡
│ 0.0       │ 40.7128      │ -74.006       │ 10.5         │ ...    │
│ 0.1       │ 40.71281     │ -74.00601     │ 10.52        │ ...    │
│ 0.2       │ 40.71282     │ -74.00602     │ 10.55        │ ...    │
└───────────┴──────────────┴───────────────┴──────────────┴────────┘
```

### Manual Synchronization (Advanced)

```python
from pils.synchronizer import CorrelationSynchronizer

# Create synchronizer
sync = CorrelationSynchronizer()

# Add reference GPS (from sensor payload)
sync.add_gps_reference(gps_df)

# Add drone GPS
if 'data' in flight.drone_data:
    sync.add_drone_gps(flight.drone_data['data'])

# Add other sensors
sync.add_payload_sensor('adc', flight['adc'].data)

# Compute synchronization
result = sync.synchronize(target_rate_hz=10.0)

print(f"Synchronized: {result.shape}")
```

---

## Step 7: Export Data

### Export to HDF5

By default the data are saved into the ```proc\``` folder, but is possible also to define a new folder 

```python
from pathlib import Path

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

# Save complete flight (includes sync_data if available)
flight.to_hdf5(output_dir / f"{flight.flight_name}.h5")
print(f"Saved raw data and synchronized data to HDF5")
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

### Reload from HDF5

```python
# Later, reload the flight
reloaded = Flight.from_hdf5(output_dir / f"{flight.flight_name}.h5")
print(f"Reloaded: {reloaded.flight_name}")
```

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
gps_df = flight['gps'].data
rtk_fixed = gps_df.filter(pl.col('fix_quality') == 4)
print(f"  Total points: {gps_df.height}")
print(f"  RTK Fixed: {rtk_fixed.height} ({rtk_fixed.height/gps_df.height:.1%})")

# Analyze IMU
print("\nIMU Analysis:")
imu = flight['imu']
accel = imu.accelerometer
print(f"  Samples: {accel.height}")
print(f"  Duration: {(accel['timestamp'].max() - accel['timestamp'].min()):.1f} seconds")

# Synchronize all data
print("\nSynchronization:")
synced_df = flight.sync(target_rate_hz=10.0)
print(f"  Synchronized: {synced_df.shape}")
print(f"  Columns: {len(synced_df.columns)}")

# Export
print("\nExporting...")
flight.to_hdf5(OUTPUT / f"{FLIGHT_ID}.h5")  # Includes sync_data
gps_df.write_parquet(OUTPUT / "gps.parquet")
synced_df.write_parquet(OUTPUT / "synchronized.parquet")
print(f"  Saved to {OUTPUT}")

print("\nDone!")
```

---

## Next Steps

- [User Guide](../user-guide/index.md) - Detailed feature documentation
- [PPK Analysis](../api/analysis/ppk.md) - Post-processed kinematics
- [API Reference](../api/index.md) - Complete API documentation
