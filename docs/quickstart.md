# Quick Start Guide

Get up and running with PILS in 5 minutes.

## Basic Workflow

```python
from pils.loader import PathLoader
from pils.flight import Flight

# 1. Load flight metadata
loader = PathLoader('/path/to/campaigns')
flight_info = loader.load_single_flight(flight_name='flight_20251208_1506')

# 2. Create flight container
flight = Flight(flight_info)

# 3. Load drone data (auto-detects platform: DJI, BlackSquare, or Litchi)
flight.add_drone_data()

# 4. Load sensor data
flight.add_sensor_data(['gps', 'imu', 'camera', 'adc'])

# 5. Access data (returns Polars DataFrames)
gps_df = flight['gps'].data
print(gps_df.head())

# 6. Save to HDF5
flight.to_hdf5('output/flight_data.h5')
```

## Example 1: GPS Analysis

```python
import polars as pl
from pils.loader import PathLoader
from pils.flight import Flight

# Load flight
loader = PathLoader('/campaigns')
flight_info = loader.load_single_flight(flight_name='my_flight')

flight = Flight(flight_info)
flight.add_sensor_data(['gps'])

# Get GPS data
gps = flight['gps'].data

# Filter by quality
high_quality = gps.filter(pl.col('fix_quality') == 4)  # RTK Fixed

# Calculate statistics
stats = high_quality.select([
    pl.col('latitude').mean().alias('mean_lat'),
    pl.col('longitude').mean().alias('mean_lon'),
    pl.col('altitude').mean().alias('mean_alt'),
    pl.col('num_satellites').mean().alias('avg_sats')
])

print(stats)
```

## Example 2: IMU Visualization

```python
import matplotlib.pyplot as plt
from pils.loader import PathLoader
from pils.flight import Flight

# Load flight with IMU
loader = PathLoader('/campaigns')
flight_info = loader.load_single_flight(flight_name='my_flight')

flight = Flight(flight_info)
flight.add_sensor_data(['imu'])

# Get accelerometer data
imu = flight['imu'].accelerometer.data

# Plot acceleration
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for idx, axis in enumerate(['ax', 'ay', 'az']):
    axes[idx].plot(imu['timestamp'], imu[axis])
    axes[idx].set_ylabel(f'{axis} (m/s²)')
    axes[idx].grid(True)

axes[-1].set_xlabel('Timestamp (ms)')
plt.tight_layout()
plt.show()
```

## Example 3: Multi-Flight Batch Processing

```python
from pathlib import Path
from pils.loader import PathLoader
from pils.flight import Flight

# Load all flights
loader = PathLoader('/campaigns')
all_flights = loader.load_all_flights()

# Process each flight
for flight_info in all_flights:
    flight_name = flight_info['flight_name']
    print(f"Processing {flight_name}...")
    
    flight = Flight(flight_info)
    flight.add_drone_data()
    flight.add_sensor_data(['gps', 'imu'])
    
    # Save to HDF5
    output_path = Path('processed') / f'{flight_name}.h5'
    output_path.parent.mkdir(exist_ok=True)
    flight.to_hdf5(output_path)
    
    print(f"  Saved to {output_path}")
```

## Example 4: PPK Analysis

```python
from pils.analyze.ppk import PPKAnalysis
from pils.loader import PathLoader
from pils.flight import Flight

# Load flight
loader = PathLoader('/campaigns')
flight_info = loader.load_single_flight(flight_name='my_flight')

flight = Flight(flight_info)
flight.add_sensor_data(['gps'])

# Initialize PPK analysis
ppk = PPKAnalysis(flight.flight_path)

# Run RTKLIB processing
version = ppk.run_analysis(
    base_station='REFERENCE_STATION',
    config_file='rtklib.conf'
)

# Access results
pos_df = version.position  # Polars DataFrame with corrected positions
stats_df = version.statistics  # Quality metrics

print(f"PPK Version: {version.name}")
print(f"Position samples: {pos_df.shape[0]}")
print(f"Mean accuracy: {stats_df['std_east'].mean():.3f} m")
```

## Example 5: Sensor Synchronization

```python
from pils.loader import PathLoader
from pils.flight import Flight
from pils.synchronizer import Synchronizer

# Load flight
loader = PathLoader('/campaigns')
flight_info = loader.load_single_flight(flight_name='my_flight')

flight = Flight(flight_info)
flight.add_drone_data()
flight.add_sensor_data(['gps', 'imu', 'inclinometer'])

# Synchronize to common time base
sync = Synchronizer()
synchronized_data = sync.synchronize(flight)

# Access synchronized data
gps_sync = synchronized_data['drone_gps']
inclino_sync = synchronized_data['inclinometer']

# All data now on same timeline
print(f"GPS samples: {gps_sync.shape[0]}")
print(f"Inclino samples: {inclino_sync.shape[0]}")
```

## Example 6: Working with Cameras

```python
from pils.loader import PathLoader
from pils.flight import Flight
import cv2

# Load flight
loader = PathLoader('/campaigns')
flight_info = loader.load_single_flight(flight_name='my_flight')

flight = Flight(flight_info)
flight.add_sensor_data(['camera'])

camera = flight['camera']

# Get frame at specific index
frame_idx = 100
frame = camera.get_frame(frame_idx)
timestamp = camera.get_timestamp(frame_idx)

print(f"Frame {frame_idx} at {timestamp}")
print(f"Shape: {frame.shape}")

# Display frame
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Common Patterns

### Loading Specific Sensors

```python
# Load only GPS and IMU
flight.add_sensor_data(['gps', 'imu'])

# Load all available sensors
flight.add_sensor_data()  # None = load all
```

### Accessing Data

```python
# Dictionary-style access
gps_data = flight['gps'].data

# Attribute access
gps_sensor = flight.raw_data['gps']
gps_data = gps_sensor.data
```

### Filtering Data

```python
import polars as pl

# Filter by time range
start_time = 1000000
end_time = 2000000
filtered = gps_df.filter(
    (pl.col('timestamp') >= start_time) & 
    (pl.col('timestamp') <= end_time)
)

# Filter by condition
high_altitude = gps_df.filter(pl.col('altitude') > 100)
```

### Saving and Loading HDF5

```python
# Save
flight.to_hdf5('data.h5')

# Load
loaded_flight = Flight.from_hdf5('data.h5')
gps_data = loaded_flight['gps'].data
```

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand system design
- Explore [Sensor Documentation](sensors.md) for detailed sensor APIs
- Check [API Reference](api_reference.md) for complete function signatures
- See [Testing Guide](testing.md) to run tests and verify installation

## Getting Help

If you encounter issues:
1. Check the [API Reference](api_reference.md)
2. Review example code in `examples/` directory
3. Run tests: `pytest tests/ -v`
4. Open an issue on GitHub
