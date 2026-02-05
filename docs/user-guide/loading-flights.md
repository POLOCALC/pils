# Loading Flights

How to discover and load flight data using PILS loaders.

## Loaders Overview

PILS provides two loaders for different use cases:

| Loader | Use Case | Data Source |
|--------|----------|-------------|
| `PathLoader` | File-based campaigns | Filesystem directories |
| `StoutLoader` | Database-backed | Stout database |

## PathLoader

The `PathLoader` discovers flights from a hierarchical filesystem structure.

### Initialization

```python
from pils.loader.path import PathLoader
from pathlib import Path

# Initialize with base path
loader = PathLoader(base_path="/data/campaigns")

# Or with Path object
loader = PathLoader(base_path=Path.home() / "data" / "campaigns")
```

### Load All Flights

```python
# Discover all flights
all_flights = loader.load_all_flights()

print(f"Found {len(all_flights)} flights")
for flight_info in all_flights[:3]:
    print(f"  {flight_info['flight_name']}")
```

**Returns:** `List[Dict[str, Any]]` - List of flight metadata dictionaries

### Load Single Flight

```python
# Load by flight ID
flight_info = loader.load_single_flight(
    flight_id="flight_20251208_1506"
)

# Or with optional campaign/date hints (faster)
flight_info = loader.load_single_flight(
    flight_id="flight_20251208_1506",
    campaign="202511",
    date="2025-12-08"
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `flight_id` | `str` | Flight identifier (folder name) |
| `campaign` | `str`, optional | Campaign ID to narrow search |
| `date` | `str`, optional | Date string (YYYY-MM-DD) |

**Returns:** `Dict[str, Any]` - Flight metadata dictionary

### Load by Date

```python
# All flights on a specific date
flights = loader.load_flights_by_date(date="2025-12-08")

# Or by campaign
flights = loader.load_flights_by_date(
    date="2025-12-08",
    campaign="202511"
)
```

### Flight Info Dictionary

The returned dictionary contains:

```python
{
    'flight_name': 'flight_20251208_1506',       # str
    'flight_path': Path('/data/.../flight_20251208_1506'),  # Path
    'date': datetime.date(2025, 12, 8),          # date
    'campaign': '202511',                         # str
    'aux_path': Path('/data/.../aux'),           # Path
    'sensors_path': Path('/data/.../sensors'),   # Path | None
    'drone_data_path': Path('/data/.../drone_data'),  # Path | None
    'camera_path': Path('/data/.../camera'),     # Path | None
}
```

---

## StoutLoader

The `StoutLoader` retrieves flight data from a Stout database.

### Initialization

```python
from pils.loader.stout import StoutLoader

# Initialize with database connection
loader = StoutLoader(db_path="/path/to/stout.db")

# Or with connection URI
loader = StoutLoader(connection_uri="sqlite:///stout.db")
```

### Usage

Same API as PathLoader:

```python
# All flights
flights = loader.load_all_flights()

# Single flight
flight_info = loader.load_single_flight(flight_id="flight_20251208_1506")

# By date
flights = loader.load_flights_by_date(date="2025-12-08")
```

---

## Creating the Flight Container

After loading flight info, create a `Flight` container:

```python
from pils.flight import Flight

# From loader
flight_info = loader.load_single_flight("flight_20251208_1506")
flight = Flight(flight_info)

# Access properties
print(flight.flight_name)   # flight_20251208_1506
print(flight.date)          # 2025-12-08
print(flight.campaign)      # 202511
print(flight.flight_path)   # /data/.../flight_20251208_1506
```

---

## Batch Processing

### Process All Flights

```python
from pils.loader.path import PathLoader
from pils.flight import Flight

loader = PathLoader(base_path="/data/campaigns")
all_flights = loader.load_all_flights()

results = []
for flight_info in all_flights:
    try:
        flight = Flight(flight_info)
        flight.add_drone_data()
        flight.add_sensor_data(['gps'])
        
        # Analyze
        gps_df = flight['gps'].data
        results.append({
            'name': flight.flight_name,
            'gps_points': gps_df.height,
            'status': 'success'
        })
    except Exception as e:
        results.append({
            'name': flight_info['flight_name'],
            'error': str(e),
            'status': 'failed'
        })

# Summary
import polars as pl
summary = pl.DataFrame(results)
print(summary)
```

### Process by Campaign

```python
# Get all flights for campaign
campaign_flights = [
    f for f in loader.load_all_flights()
    if f['campaign'] == '202511'
]

for flight_info in campaign_flights:
    process_flight(flight_info)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor
from pils.loader.path import PathLoader
from pils.flight import Flight

def process_flight(flight_info):
    """Process single flight (runs in separate process)."""
    flight = Flight(flight_info)
    flight.add_drone_data()
    flight.add_sensor_data(['gps', 'imu'])
    flight.to_hdf5(f"output/{flight.flight_name}.h5")
    return flight.flight_name

# Load flight info (serializable)
loader = PathLoader(base_path="/data/campaigns")
all_flights = loader.load_all_flights()

# Process in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_flight, all_flights))

print(f"Processed {len(results)} flights")
```

---

## Error Handling

### Flight Not Found

```python
from pils.loader.path import PathLoader

loader = PathLoader(base_path="/data/campaigns")

try:
    flight_info = loader.load_single_flight("nonexistent_flight")
except FileNotFoundError as e:
    print(f"Flight not found: {e}")
```

### Invalid Directory Structure

```python
try:
    loader = PathLoader(base_path="/invalid/path")
    flights = loader.load_all_flights()
except FileNotFoundError:
    print("Base path does not exist")
except ValueError as e:
    print(f"Invalid directory structure: {e}")
```

### Missing Required Data

```python
flight = Flight(flight_info)

try:
    flight.add_drone_data()
except FileNotFoundError:
    print("No drone data found")
except ValueError as e:
    print(f"Unsupported drone platform: {e}")
```

---

## Best Practices

!!! tip "Use Specific Loaders"
    Choose `PathLoader` for filesystem data, `StoutLoader` for database data.

!!! tip "Provide Hints"
    When loading a single flight, provide `campaign` and `date` hints for faster lookup.

!!! tip "Handle Errors"
    Always wrap loading in try/except for batch processing.

!!! tip "Use Lazy Loading"
    Don't load all sensors if you only need GPS. Use `flight.add_sensor_data(['gps'])`.

---

## Next Steps

- [Working with Sensors](sensors.md) - Access sensor data
- [Drone Platforms](drones.md) - Platform-specific features
