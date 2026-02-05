# Loaders

Flight discovery and loading modules.

## Overview

Loaders discover and load flight metadata from various sources:

| Loader | Source | Use Case |
|--------|--------|----------|
| [PathLoader](path-loader.md) | Filesystem | File-based campaigns |
| [StoutLoader](stout-loader.md) | Database | Database-backed storage |

## Common API

Both loaders share the same API:

```python
# Load all flights
flights = loader.load_all_flights() -> List[Dict[str, Any]]

# Load single flight
flight_info = loader.load_single_flight(
    flight_id: str,
    campaign: Optional[str] = None,
    date: Optional[str] = None
) -> Dict[str, Any]

# Load by date
flights = loader.load_flights_by_date(
    date: str,
    campaign: Optional[str] = None
) -> List[Dict[str, Any]]
```

## Flight Info Dictionary

All loaders return dictionaries with this structure:

```python
{
    'flight_name': str,        # e.g., "flight_20251208_1506"
    'flight_path': Path,       # Absolute path to flight directory
    'date': datetime.date,     # Flight date
    'campaign': str,           # Campaign identifier
    'aux_path': Path,          # Path to aux/ directory
    'sensors_path': Path,      # Path to sensors/ (optional)
    'drone_data_path': Path,   # Path to drone_data/ (optional)
    'camera_path': Path,       # Path to camera/ (optional)
}
```

## Quick Start

```python
from pils.loader.path import PathLoader
from pils.flight import Flight

# Initialize
loader = PathLoader(base_path="/data/campaigns")

# Discover
all_flights = loader.load_all_flights()
print(f"Found {len(all_flights)} flights")

# Load one
flight_info = loader.load_single_flight("flight_20251208_1506")

# Create container
flight = Flight(flight_info)
```

## Detailed Documentation

- [PathLoader](path-loader.md) - Filesystem-based loading
- [StoutLoader](stout-loader.md) - Database-backed loading
