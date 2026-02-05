# User Guide

Comprehensive guide to all PILS features and capabilities.

## Overview

PILS provides a complete workflow for the POLOCALC data processing:

```mermaid
flowchart TB
    subgraph Loading
        A[PathLoader] --> C[Flight Container]
        B[StoutLoader] --> C
    end
    
    subgraph Data
        C --> D[Drone Data]
        C --> E[Sensor Data]
    end
    
    subgraph Processing
        D --> F[Synchronizer]
        E --> F
        F --> G[Synchronized Data]
    end
    
    subgraph Analysis
        subgraph PPK Analysis
            A --> H[PPK Analysis]
            B --> H[PPK Analysis]
        end
        G --> J[Custom Analysis]
    end
    
    subgraph Export
        H --> K[HDF5]
        H --> I[Markdown Report]
        J --> L[CSV/Parquet]
    end
```

## Chapters

<div class="grid cards" markdown>

-   :material-folder-open:{ .lg .middle } __[Directory Structure](directory-structure.md)__

    ---

    Explain the directory and file structure for PILS

-   :material-radar:{ .lg .middle } __[Time Synchronization](synchronization.md)__

    ---

    Mathematical description of the syncrhonization process between the different dataset

-   :material-quadcopter:{ .lg .middle } __[DJI Data](dji-drone.md)__

    ---

    Description for handling the data from the DJI Matrice 600

</div>

## Quick Reference

### Loading

```python
from pils.loader.path import PathLoader
from pils.flight import Flight

loader = PathLoader(base_path="/campaigns")
flight_info = loader.load_single_flight("flight_20251208_1506")
flight = Flight(flight_info)
```

### Accessing Data

```python
# Drone data
flight.add_drone_data()
drone_df = flight.drone_data['data']

# Sensors
flight.add_sensor_data(['gps', 'imu'])
gps_df = flight['gps'].data
accel_df = flight['imu'].accelerometer
```

### Analysis

```python
from pils.synchronizer import Synchronizer

sync = Synchronizer()
sync.add_gps_reference(gps_df)
sync.add_drone_gps(drone_df)
result = sync.synchronize()
```

### Export

```python
# Complete flight to HDF5
flight.to_hdf5("flight.h5")

# Individual DataFrame
gps_df.write_parquet("gps.parquet")
```
