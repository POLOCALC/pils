# Architecture

PILS system design and component overview.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A[Local Files]
        B[Remote Storage]
    end
    
    subgraph "Loaders"
        C[PathLoader]
        D[StoutLoader]
    end
    
    subgraph "Core"
        E[Flight Container]
        F[Synchronizer]
    end
    
    subgraph "Data Types"
        G[Sensors]
        H[Drones]
    end
    
    subgraph "Analysis"
        I[PPK Processing]
        J[RTK Analysis]
    end
    
    subgraph "Output"
        K[HDF5]
        L[CSV]
        M[Parquet]
    end
    
    A --> C
    B --> D
    C --> E
    D --> E
    G --> E
    H --> E
    E --> F
    F --> E
    E --> I
    E --> J
    E --> K
    E --> L
    E --> M
```

---

## Class Hierarchy

### Core Classes

```mermaid
classDiagram
    class Flight {
        +flight_info: Dict
        +raw_data: Dict
        +synchronized: DataFrame
        +add_sensor_data()
        +add_drone_data()
        +to_hdf5()
        +from_hdf5()
    }
    
    class Loader {
        <<abstract>>
        +load_single_flight()
        +list_flights()
    }
    
    class PathLoader {
        +data_path: Path
        +load_single_flight()
    }
    
    class StoutLoader {
        +api_endpoint: str
        +load_single_flight()
    }
    
    Loader <|-- PathLoader
    Loader <|-- StoutLoader
    Loader ..> Flight
```

### Sensor Classes

```mermaid
classDiagram
    class BaseSensor {
        <<abstract>>
        +file: Path
        +data: DataFrame
        +load_data()
    }
    
    class GPS {
        +parse_nmea()
        +parse_ubx()
    }
    
    class IMU {
        +compute_euler()
    }
    
    class ADC {
        +calibrate()
    }
    
    class Camera {
        +parse_events()
    }
    
    class Inclinometer {
        +decode_kernel()
    }
    
    BaseSensor <|-- GPS
    BaseSensor <|-- IMU
    BaseSensor <|-- ADC
    BaseSensor <|-- Camera
    BaseSensor <|-- Inclinometer
```

### Drone Classes

```mermaid
classDiagram
    class BaseDrone {
        <<abstract>>
        +file: Path
        +data: DataFrame
        +load_data()
    }
    
    class DJIDrone {
        +parse_srt()
        +parse_csv()
    }
    
    class Litchi {
        +parse_log()
    }
    
    class BlackSquareDrone {
        +parse_telemetry()
    }
    
    BaseDrone <|-- DJIDrone
    BaseDrone <|-- Litchi
    BaseDrone <|-- BlackSquareDrone
```

---

## Data Flow

### Loading Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Loader
    participant Flight
    participant Sensor
    
    User->>Loader: load_single_flight(path)
    Loader->>Flight: create Flight()
    Loader->>Flight: set flight_info
    User->>Flight: add_sensor_data(['gps'])
    Flight->>Sensor: GPS(file)
    Sensor->>Sensor: load_data()
    Sensor-->>Flight: return sensor
    Flight->>Flight: store in raw_data
```

### Synchronization Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Sync as Synchronizer
    participant Flight
    
    User->>Sync: synchronize(flight)
    Sync->>Flight: get raw_data
    Sync->>Sync: compute correlations
    Sync->>Sync: align timestamps
    Sync->>Sync: merge DataFrames
    Sync->>Flight: store synchronized
    Sync-->>User: return flight
```

---

## Component Details

### Flight Container

Central data structure holding all flight information:

```python
flight = Flight(flight_info={...})
flight.raw_data['gps'] = GPS(...)
flight.raw_data['imu'] = IMU(...)
flight.synchronized = merged_df
```

**Responsibilities:**

- Store flight metadata
- Manage raw sensor data
- Store synchronized data versions
- Handle HDF5 serialization

### Loaders

Abstract data source interaction:

| Loader | Source | Features |
|--------|--------|----------|
| PathLoader | Local filesystem | Recursive search, auto-detect |
| StoutLoader | POLOCALC STOUT | API access, caching |

### Sensors

Sensor-specific data handling:

| Sensor | Formats | Features |
|--------|---------|----------|
| GPS | CSV, NMEA, UBX | Coordinate parsing |
| IMU | CSV | Euler computation |
| ADC | CSV | Calibration |
| Camera | CSV | Event parsing |
| Inclinometer | Binary | KERNEL decode |

### Synchronizer

Time alignment of multi-sensor data:

```mermaid
graph LR
    A[GPS timestamps] --> D[Correlation]
    B[IMU timestamps] --> D
    C[Drone timestamps] --> D
    D --> E[Time offsets]
    E --> F[Aligned merge]
```

---

## Design Patterns

### Factory Pattern

Sensor creation via registry:

```python
sensor_config = {
    'gps': {'class': GPS, 'load_method': 'load_data'},
    'imu': {'class': IMU, 'load_method': 'load_data'},
}

def create_sensor(sensor_type: str, file: Path):
    config = sensor_config[sensor_type]
    return config['class'](file=file)
```

### Strategy Pattern

Synchronization methods:

```python
class CorrelationSynchronizer:
    def synchronize(self, flight): ...

class InterpolationSynchronizer:
    def synchronize(self, flight): ...
```

### Repository Pattern

Data access abstraction:

```python
class PathLoader:
    def load_single_flight(self, path): ...

class StoutLoader:
    def load_single_flight(self, flight_id): ...
```

---

## Module Dependencies

```mermaid
graph TD
    pils.flight --> pils.sensors
    pils.flight --> pils.drones
    pils.flight --> pils.synchronizer
    pils.loader.path --> pils.flight
    pils.loader.stout --> pils.flight
    pils.analyze.ppk --> pils.flight
    pils.sensors --> pils.utils
    pils.drones --> pils.utils
```

---

## See Also

- [API Reference](../api/index.md) - Detailed API docs
- [Adding Sensors](adding-sensors.md) - Extend with new sensors
- [Code Style](code-style.md) - Coding standards
