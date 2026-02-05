# HDF5 Format

Hierarchical Data Format 5 specifications for PILS flight data.

## Overview

PILS uses HDF5 for efficient storage and retrieval of flight data. The format supports:

- **Hierarchical structure** - Organized groups and datasets
- **Compression** - GZIP compression for efficient storage
- **Metadata** - Attributes for flight info and processing history
- **Versioned sync** - Multiple synchronized data versions

---

## File Structure

```
flight.h5
├── metadata/
│   ├── flight_info/                    # Flight identification
│   │   ├── flight_id: "flight-001"
│   │   ├── date: "2023-11-01"
│   │   └── ...
│   └── flight_metadata/                # Processing metadata
│       ├── pils_version: "1.0.0"
│       ├── created_at: "2023-11-01T12:00:00"
│       └── ...
├── raw_data/
│   ├── drone/                          # Drone telemetry
│   │   └── [DataFrame columns]
│   ├── litchi/                         # Litchi data (if present)
│   │   └── [DataFrame columns]
│   └── sensors/
│       ├── gps/
│       │   └── [DataFrame columns]
│       ├── imu/
│       │   └── [DataFrame columns]
│       └── ...
└── synchronized/                        # Versioned sync data
    ├── rev_20231101_120000/
    │   ├── @timestamp_col: "timestamp"
    │   ├── @sync_method: "correlation"
    │   └── [Merged DataFrame columns]
    └── rev_20231101_140000/
        └── ...
```

---

## Groups

### metadata/flight_info

Flight identification attributes.

| Attribute | Type | Description |
|-----------|------|-------------|
| `flight_id` | `str` | Unique flight identifier |
| `date` | `str` | Flight date (YYYY-MM-DD) |
| `platform` | `str` | Drone platform type |
| `pilot` | `str` | Pilot name |
| `location` | `str` | Flight location |
| `notes` | `str` | Additional notes |

### metadata/flight_metadata

Processing metadata.

| Attribute | Type | Description |
|-----------|------|-------------|
| `pils_version` | `str` | PILS version used |
| `created_at` | `str` | Creation timestamp |
| `modified_at` | `str` | Last modification |
| `data_sources` | `json` | Source file paths |

### raw_data/drone

Drone telemetry data stored as columnar datasets.

### raw_data/sensors/[sensor_name]

Sensor data stored as columnar datasets.

### synchronized/[version_id]

Merged synchronized data with version-specific attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `timestamp_col` | `str` | Reference timestamp column |
| `sync_method` | `str` | Synchronization method |
| `sources` | `json` | Source columns mapping |
| `created_at` | `str` | Sync timestamp |

---

## DataFrame Storage

Polars DataFrames are stored column-by-column:

```python
# Storage structure
dataset_group/
├── column_name_1: [values...]
├── column_name_2: [values...]
└── column_name_3: [values...]
```

### Column Type Mapping

| Polars Type | HDF5 Type | Notes |
|-------------|-----------|-------|
| `Int64` | `int64` | - |
| `Int32` | `int32` | - |
| `Float64` | `float64` | - |
| `Float32` | `float32` | - |
| `Boolean` | `bool` | - |
| `Utf8` | `str` (special) | Variable-length |
| `Datetime` | `int64` | Microseconds |
| `Date` | `int32` | Days since epoch |
| `List` | `json str` | Serialized |

---

## Reading HDF5

### Using Flight Class

```python
from pils.flight import Flight

# Load with latest sync version
flight = Flight.from_hdf5('flight_001.h5')

# Load specific sync version
flight = Flight.from_hdf5('flight_001.h5', sync_version='rev_20231101_120000')

# Load without sync data
flight = Flight.from_hdf5('flight_001.h5', sync_version=False)
```

### Direct h5py Access

```python
import h5py
import polars as pl

with h5py.File('flight.h5', 'r') as f:
    # Read metadata
    flight_id = f['metadata/flight_info'].attrs['flight_id']
    
    # Read GPS data
    gps_group = f['raw_data/sensors/gps']
    gps_data = {col: gps_group[col][:] for col in gps_group.keys()}
    gps_df = pl.DataFrame(gps_data)
    
    # List sync versions
    if 'synchronized' in f:
        versions = list(f['synchronized'].keys())
        print(f"Sync versions: {versions}")
```

---

## Writing HDF5

### Using Flight Class

```python
from pils.flight import Flight

# Save flight data
flight.to_hdf5('flight_001.h5')

# Overwrite existing
flight.to_hdf5('flight_001.h5', overwrite=True)
```

### Direct h5py Access

```python
import h5py
import polars as pl

df = pl.DataFrame({
    'timestamp': [1700000000000000, 1700000001000000],
    'lat': [40.7128, 40.7129],
    'lon': [-74.0060, -74.0061],
})

with h5py.File('data.h5', 'w') as f:
    grp = f.create_group('gps')
    for col in df.columns:
        grp.create_dataset(
            col,
            data=df[col].to_numpy(),
            compression='gzip',
            compression_opts=4
        )
```

---

## Compression

PILS uses GZIP compression by default:

| Setting | Value |
|---------|-------|
| Algorithm | GZIP |
| Level | 4 (balanced) |
| Chunk size | Auto |

### Compression Comparison

| Level | Ratio | Speed |
|-------|-------|-------|
| 0 | 1.0x | Fastest |
| 4 | 3-5x | Balanced |
| 9 | 5-7x | Slowest |

---

## Example: Full Workflow

```python
from pils.flight import Flight
from pils.loader import PathLoader
from pils.synchronizer import CorrelationSynchronizer
import polars as pl

# Load flight
loader = PathLoader()
flight = loader.load_single_flight(data_path="/data/flight-001")

# Add sensor data
flight.add_sensor_data(['gps', 'imu'])

# Synchronize
sync = CorrelationSynchronizer()
flight = sync.synchronize(flight)

# Save to HDF5
flight.to_hdf5("flight-001.h5")

# Later: reload
loaded = Flight.from_hdf5("flight-001.h5")

# Access data
print(loaded['gps'].data.head())
print(loaded.synchronized.head())
```

---

## Inspection Tools

### h5dump (CLI)

```bash
# Show structure
h5dump -H flight.h5

# Show specific group
h5dump -H -g /raw_data/sensors/gps flight.h5
```

### HDFView (GUI)

Download from: https://www.hdfgroup.org/downloads/hdfview/

### Python inspection

```python
import h5py

def print_hdf5_structure(name, obj):
    """Print HDF5 file structure."""
    indent = '  ' * name.count('/')
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}{name}: {obj.shape} {obj.dtype}")
    else:
        print(f"{indent}{name}/")
        for key, value in obj.attrs.items():
            print(f"{indent}  @{key}: {value}")

with h5py.File('flight.h5', 'r') as f:
    f.visititems(print_hdf5_structure)
```

---

## See Also

- [Flight API](../api/core/flight.md) - Flight class documentation
- [Binary Formats](binary.md) - Binary file formats
- [Text Formats](text.md) - CSV and ASCII formats
