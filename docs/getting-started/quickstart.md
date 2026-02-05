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
# Access GPS data
gps = flight['gps']
gps_df = gps.data  # Polars DataFrame

# Filter and analyze
import polars as pl

high_quality = gps_df.filter(pl.col('fix_quality') >= 4)
print(f"RTK fixes: {high_quality.height}")
```

## Step 5: Synchronize Data (Optional)

```python
# Synchronize all sensors to common time base
synced_df = flight.sync(target_rate_hz=10.0)

print(f"Synchronized: {synced_df.shape}")
print(f"Time range: {synced_df['timestamp'].min():.1f}s - {synced_df['timestamp'].max():.1f}s")
```

## Step 6: Export Results

RAW data can be exported into a single .h5 file which contains also the synchronized data

```python
# Export to HDF5 (includes sync_data if available)
flight.to_hdf5("/path/to/output/flight.h5")
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

# Analyze GPS
gps_df = flight['gps'].data
print(f"Total points: {gps_df.height}")
print(f"RTK fixes: {gps_df.filter(pl.col('fix_quality') == 4).height}")

# Analyze IMU
imu = flight['imu']
accel_df = imu.accelerometer
print(f"IMU samples: {accel_df.height}")
print(f"Max acceleration: {accel_df['az'].max():.2f} m/s²")

# Synchronize all data
synced_df = flight.sync(target_rate_hz=10.0)
print(f"Synchronized: {synced_df.shape}")

# Save results
output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

flight.to_hdf5(output_dir / f"{FLIGHT_ID}.h5")
print(f"Saved to {output_dir}")
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
synced_df = flight.sync(target_rate_hz=10.0)
print(f"Synchronized data: {synced_df.shape}")

# Synchronized data is stored in flight.sync_data
print(f"Columns: {flight.sync_data.columns}")

# Option 2: Manual synchronization with CorrelationSynchronizer
from pils.synchronizer import CorrelationSynchronizer

sync = CorrelationSynchronizer()
sync.add_gps_reference(flight['gps'].data)
sync.add_drone_gps(flight.drone_data['data'])

result = sync.synchronize(target_rate_hz=10.0)
print(f"Synchronized data: {result.shape}")
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
