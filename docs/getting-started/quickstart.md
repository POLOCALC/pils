# Quick Start

Get up and running with PILS in 5 minutes.

## Basic Workflow

```mermaid
graph LR
    A[Load Flight] --> B[Add Drone Data]
    B --> C[Add Sensors]
    C --> D[Analyze]
    D --> E[Export]
```

## Step 1: Load a Flight

=== "PathLoader"

    ```python
    from pils.loader.path import PathLoader
    
    # Initialize loader with campaigns directory
    loader = PathLoader(base_path="/path/to/campaigns")
    
    # Load specific flight
    flight_info = loader.load_single_flight(
        flight_id="flight_20251208_1506"
    )
    ```

=== "StoutLoader"

    ```python
    from pils.loader.stout import StoutLoader
    
    # Initialize loader with database connection
    loader = StoutLoader(db_path="/path/to/stout.db")
    
    # Load flight by ID
    flight_info = loader.load_single_flight(
        flight_id="flight_20251208_1506"
    )
    ```

## Step 2: Create Flight Container

```python
from pils.flight import Flight

# Create flight container
flight = Flight(flight_info)

# Check flight metadata
print(f"Flight: {flight.flight_name}")
print(f"Date: {flight.date}")
print(f"Path: {flight.flight_path}")
```

## Step 3: Load Data

```python
# Load drone platform data (auto-detects DJI/ArduPilot/Litchi)
flight.add_drone_data()

# Load specific sensors
flight.add_sensor_data(['gps', 'imu'])

# Or load all available sensors
flight.add_sensor_data()
```

## Step 4: Access Data

```python
import polars as pl

payload = flight.raw_data.payload_data

# Payload GPS: lat_deg, lon_deg, h_ell_m, h_msl_m, h_acc_m, ...
gps_df = payload.gps
good = gps_df.filter(pl.col("h_acc_m") < 2.0)
print(f"Samples with horizontal accuracy < 2 m: {good.height}")
```

Column names carry their unit (`_deg`, `_m`, `_g`, ...); see
[Data Schema](../user-guide/data-schema.md).

## Step 5: Synchronize Data (Optional)

```python
# Synchronize all sources on the payload GPS time base
synced = flight.sync(target_rate={"drone": 10.0, "payload": 10.0})

print(list(synced))            # drone, litchi, inclinometer, adc, ...
print(synced["drone"].select("timestamp", "lat_deg", "lon_deg", "h_ell_m"))
```

## Step 6: Save Results

`Flight.save` writes the raw data and a new synchronized revision as Parquet
tables with a manifest; `Flight.load` reads them back.

```python
flight.save("/path/to/output/flight")
reloaded = Flight.load("/path/to/output/flight")
```

---

## Complete Example

```python
from pathlib import Path
import polars as pl

from pils.loader.path import PathLoader
from pils.flight import Flight

# Configuration
CAMPAIGNS = Path("/data/campaigns")
FLIGHT_ID = "flight_20251208_1506"

# Load flight
loader = PathLoader(base_path=CAMPAIGNS)
flight_info = loader.load_single_flight(flight_id=FLIGHT_ID)

# Create container and load data
flight = Flight(flight_info)
flight.add_drone_data()
flight.add_sensor_data(['gps', 'imu'])
payload = flight.raw_data.payload_data

# Analyze GPS
print(f"Total points: {payload.gps.height}")

# Analyze IMU
imu = payload.imu  # {"barometer": df, "accelerometer": df, ...}
accel_df = imu['accelerometer']
print(f"IMU samples: {accel_df.height}")
print(f"Max acceleration: {accel_df['acc_z_g'].max():.2f} g")

# Synchronize all data
synced = flight.sync(target_rate={"drone": 10.0, "payload": 10.0})
print(f"Synchronized drone: {synced['drone'].shape}")

# Save results
flight.save(Path("./output") / FLIGHT_ID)
```

---

## Common Patterns

### Batch Processing

```python
# Load all flights in a date
flights = loader.load_flights_by_date(date="2025-12-08")

for flight_info in flights:
    flight = Flight(flight_info)
    flight.add_drone_data()
    flight.add_sensor_data()
    
    # Process each flight
    process_flight(flight)
```

### Time Synchronization

```python
# Option 1: Use Flight.sync() - recommended for most cases
synced = flight.sync(target_rate={"drone": 10.0})

# Synchronized data is stored in flight.sync_data, one table per source
print({name: df.shape for name, df in flight.sync_data.items()})

# Option 2: Manual synchronization with the Synchronizer
from pils.synchronizer import Synchronizer

sync = Synchronizer()
sync.add_gps_reference(flight.raw_data.payload_data.gps)
sync.add_drone_gps(
    flight.raw_data.drone_data.drone,
    lat_col="rtk_lat_deg",
    lon_col="rtk_lon_deg",
    alt_col="rtk_h_ell_m",
)

result = sync.synchronize(target_rate={"drone": 10.0})
print(sync.get_offset_summary())
```

<!-- ### PPK Analysis

```python
from pils.analyze.ppk import PPKAnalyzer

ppk = PPKAnalyzer(flight_path=flight.flight_path)
version = ppk.process(
    rover_obs="rover.obs",
    base_obs="base.obs",
    nav="rover.nav"
)

print(f"Fix rate: {version.fix_rate:.1%}")
print(f"Mean accuracy: {version.mean_accuracy:.3f} m")
```

--- -->

## Next Steps

- [First Flight Tutorial](first-flight.md) - Detailed walkthrough
- [User Guide](../user-guide/index.md) - Complete feature documentation
- [API Reference](../api/index.md) - Full API documentation
