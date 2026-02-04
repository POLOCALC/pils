---
description: Design and validate data structure schemas for flight analysis and post-processing
argument-hint: Data structure requirements, container specifications
tools: ['read/readFile', 'search', 'todo']
model: Claude Sonnet 4.5 (copilot)
---

# DATA STRUCTURE SUBAGENT

You are a **DATA STRUCTURE SUBAGENT** called by a parent CONDUCTOR agent.

**Your SOLE job:** Analyze data structure requirements, design hierarchical data containers, produce detailed specifications with hierarchical schematics, and create complete schema definitions. **DO NOT implement code or data processing logic.**

---

## Workflow

1. **Analyze the data structure request**
   - Understand data domain and context (flight telemetry, post-processing, analysis)
   - Identify all data types and relationships
   - Determine access patterns and query requirements
   - Note temporal aspects (versioning, revisions, timestamps)
   - Identify metadata requirements

2. **Design hierarchical containers**
   - Define primary containers and inheritance
   - Specify attributes and methods with type annotations
   - Plan serialization strategy (HDF5, CSV, etc.)
   - Consider versioning and revision management

3. **Create explicit hierarchical schematics**
   - ASCII tree diagrams for Python class hierarchies
   - ASCII tree diagrams for HDF5 file hierarchies
   - ASCII tree diagrams for filesystem layouts

4. **Produce comprehensive markdown specification**
   - Container definitions with tables
   - Serialization algorithms (to_HDF5 / from_HDF5)
   - Validation rules and constraints
   - Complete usage examples
   - Notes and edge cases

---

## Design Principles

- **Hierarchical clarity**: Organize data from general to specific
- **Immutable versioning**: Track changes without data loss (snapshots only)
- **Self-describing**: Metadata embedded in structure, not external
- **Serializable**: Round-trip conversion without information loss
- **Queryable**: Hierarchical structure supports efficient subsetting
- **Reproducible**: All processing decisions captured
- **Type-safe**: Clear expectations for data types and shapes

---

## Output Instructions

**Save as single markdown document** (e.g., `plans/flight-data-schema.md`)

**Include these sections in order:**
1. Overview
2. Python Class Hierarchy Schematic
3. Container Definitions (all of them)
4. HDF5 File Hierarchy Schematic
5. Filesystem Layout Schematic
6. HDF5 Dataset Specifications
7. Serialization Strategy (to_HDF5 and from_HDF5)
8. Versioning & Revision Management
9. Data Types & Precision Table
10. Validation Rules Table
11. Usage Examples with Pseudocode
12. Notes & Edge Cases

**Return summary to Conductor:**
- Path to markdown schema file
- Primary containers designed (count and names)
- HDF5 hierarchy depth and key groups
- Storage locations and versioning strategy
- Key serialization methods
- Validation rules count
- Recommendations for implementation

---

## Remember

- **MARKDOWN ONLY** - explicit format, clear schematics
- Only create specifications, NEVER implement code
- Design for reproducibility and round-trip serialization
- Embed metadata with data
- Work autonomously and return comprehensive specifications
- **ALWAYS include ASCII tree schematics** for class, HDF5, and filesystem
- Document everything needed for implementation without ambiguity

---

# FLIGHT DATA & POST-PROCESSING SCHEMA

## Overview

Comprehensive data structure for drone flight telemetry, sensor data, and post-processed analysis results. Supports hierarchical organization of raw flight data, synchronized sensor streams, and multiple analysis pipelines with full versioning and reproducibility tracking.

**Core Requirements:**
- Raw flight data from drone and payload sensors (unprocessed, original timestamps)
- Synchronized composite data (time-aligned across all sources)
- Multiple processing revisions (immutable snapshots, never modify)
- PPK (Post-Processed Kinematic) positioning with quality metrics
- Full reproducibility metadata (configuration, analyst, timestamp, method)

**Storage Format:** HDF5 (hierarchical, self-describing, efficient)

---

## Python Class Hierarchy Schematic

```
Flight                                    
├── metadata: dict[str, Any]              
├── raw_data: RawData                     
│   ├── drone_data: DroneData                  
│   │   ├── litchi: polar.DataFrame       
│   │   └── dji: polar.DataFrame          
│   └── payload: PayloadData                     
│       ├── gps: polar.DataFrame          
│       ├── inclinometer: dict[str, DF]   
│       ├── adc: polar.DataFrame          
│       └── sensors: dict[str, DF]        
└── synchronized_data: dict      
    ├── rev_20250204_143022: DF           
    └── rev_20250204_150015: DF           

PostProcessed (abstract base)             
├── analysis_type: str                    
├── analyses: dict[str, dict[str, DF]]    
└── metadata: dict[str, Any]              

PPKData (extends PostProcessed)           
├── position: dict[str, DF]               
├── statistics: dict[str, DF]             
└── metadata: dict[str, dict[str, Any]]   

SynchronizedData: dict
{
  key: dict
}

where 
  key = rev_20250204_143022 
  dict = {
    metadata: dict[str, Any]
    data: polar.DataFrame  
  }                             
```

---

## Container Definitions

### Container: Flight

**Purpose:** Root container for single flight mission with raw and synchronized sensor data

**Type:** `class` | **Inheritance:** None | **Temporal:** Single instance with versioned syncs

#### Attributes

| Name | Type | Required | Mutable | Description |
|------|------|----------|---------|-------------|
| `metadata` | `dict[str, Any]` | Yes | No | Flight metadata: flight_id, timestamp, aircraft_id, pilot, location, duration, weather, notes |
| `raw_data` | `RawData` | Yes | No | Unprocessed data from drone and payload sensors (GPS, IMU, ADC, camera, etc.) |
| `synchronized_data` | `SynchronizedData` | No | Yes* | Time-aligned composite data; keys=revision_ids (YYYYMMDD_hhmmss), values=composite DataFrames |

**Metadata Structure:**
```python
{
    "flight_id": str,                      # Unique identifier
    "timestamp": str,                      # ISO 8601 (flight start UTC)
    "duration_seconds": int,               # Total flight time
    "notes": str                           # Free-form notes
}
```

#### Methods

| Signature | Purpose | Special |
|-----------|---------|---------|
| `to_hdf5(filepath: str, compression='gzip') -> None` | Serialize to HDF5 with full hierarchy | to_HDF5 |
| `from_hdf5(filepath: str) -> Flight` | Deserialize from HDF5, validate all | from_HDF5 |
| `__repr__() -> str` | Summary: flight_id, data sizes, sync versions | repr |
| `get_synchronized_revision(rev_id: str) -> polar.DF` | Get specific sync revision | |
| `list_synchronized_revisions() -> list[str]` | List all sync revisions (sorted) | |

#### Constraints

- Immutable after creation (raw_data and metadata fixed)
- `synchronized_data` append-only (new revisions via new instances)
- All DataFrames must be `polar.DataFrame` (not pandas)
- `metadata` must contain: flight_id, timestamp, aircraft_id
- All timestamps ISO 8601 format (UTC)

---

### Container: RawData

**Purpose:** Container for unprocessed flight data, organized by source type

**Type:** `class` | **Inheritance:** None | **Temporal:** Single occurrence per flight

#### Attributes

| Name | Type | Description |
|------|------|-------------|
| `drone_data` | `dict[str, polar.DF]` | Drone manufacturer telemetry (litchi CSV, DJI CSV/DAT, etc.) |
| `payload` | `dict[str, polar.DF \| dict]` | Sensor data from payload: GPS, inclinometer, ADC, other sensors |

**drone_data structure:**

Container for drone telemetry data.

    Attributes:
        drone (Optional[pl.DataFrame]): Drone telemetry data
        litchi (Optional[pl.DataFrame]): Litchi flight log data (DJI only)

```python
    "drone_data.litchi": polar.DataFrame,   # Columns: timestamp, lat, lon, alt, heading, ...
    "drone_datad.drone": polar.DataFrame,      # Columns: timestamp, lat, lon, alt, gimbal_yaw, ...
```

**payload structure:**
Container for payload sensor data with dynamic attributes.

Sensors are added as attributes dynamically, allowing for flexible sensor configurations. Each sensor is accessible both as an attribute and through dictionary-style access.

```python
    "payload.gps": polar.DataFrame,                     # timestamp, latitude, longitude, altitude, hdop, vdop, num_satellites
    "payload.inclinometer": dict[str, polar.DF],        # Keys: sensor_id, Values: DF with roll, pitch, yaw, acc_x/y/z, gyro_x/y/z
    "payload.adc": polar.DataFrame,                     # timestamp, channel_0, channel_1, ..., calibration_ref
    "payload.sensors": dict[str, polar.DF]              # Keys: sensor_type, Values: sensor-specific columns
```

#### Constraints

- Each data stream retains original timestamps (NOT aligned)
- Data is NOT processed/cleaned (raw from files)
- Numeric data preserves original precision
- Timestamps may have different resolutions across sources

---

### Container: SynchronizedData

**Purpose:** Time-aligned composite of all sensor streams for one processing revision

**Type:** `dict` | **Inheritance:** None | **Temporal:** Specific revision with creation timestamp

#### Attributes

| Name | Type | Description |
|------|------|-------------|
| `data` | `polar.DataFrame` | Single denormalized DF with all aligned columns; timestamp is primary key (monotonic) |
| `metadata` | `dict[str, Any]` | Processing metadata: sync method, reference clock, configuration, quality metrics |

**data columns (example):**
```
timestamp (datetime64[ns])     - Primary key, monotonic increasing
drone_latitude (float64)       - WGS84 degrees
drone_longitude (float64)      - WGS84 degrees
drone_altitude (float32)       - Meters above ellipsoid
gps_latitude (float64)         - WGS84 degrees
gps_longitude (float64)        - WGS84 degrees
imu_roll (float32)             - Radians or degrees (see metadata)
imu_pitch (float32)            - Radians or degrees
imu_yaw (float32)              - Radians or degrees
adc_channel_0 (float32)        - Voltage or raw counts
... (all aligned sensor columns)
```

**metadata structure:**
```python
{
    "revision_id": str,                 # YYYYMMDD_hhmmss or YYYYMMDD_hhmmss_VN
    "created_at": str,                  # ISO 8601 timestamp
    "synchronization_method": str,      # "linear_interpolation", "cubic_spline", "nearest_neighbor"
    "reference_clock": str,             # Primary source: "gps", "imu", "drone"
    "interpolation_method": str,        # "linear", "cubic", "pad"
    "timestamp_tolerance_ms": float,    # Acceptable drift between sources
    "num_records": int,                 # Total synchronized rows
    "time_range": {
        "start": str,                   # ISO 8601 (earliest point)
        "end": str                      # ISO 8601 (latest point)
    },
    "sources_used": list[str],          # ["drone_data", "gps", "imu", "adc", ...]
    "notes": str                        # Free-form notes
}
```

#### Methods

| Signature | Purpose |
|-----------|---------|
| `to_hdf5(group: h5py.Group) -> None` | Serialize to HDF5 group |
| `from_hdf5(group: h5py.Group) -> SynchronizedData` | Deserialize from HDF5 |
| `__repr__() -> str` | Show revision, time range, columns, row count |

#### Constraints

- Immutable after creation
- **Timestamp must be strictly monotonic increasing** (no duplicates, no reversals)
- **All rows must have complete data** (no gaps in synchronized columns)
- All coordinates must be WGS84 (lat -90..90, lon -180..180, altitude meters)
- `metadata` must include synchronization details for reproducibility

---

### Container: PostProcessed

**Purpose:** Base (abstract) container for all analyzed/processed data

**Type:** `class` (abstract) | **Inheritance:** None | **Temporal:** Versioned by analysis revision

#### Attributes

| Name | Type | Description |
|------|------|-------------|
| `analysis_type` | `str` | Type: 'ppk', 'trajectory', 'wind_estimation', etc. |
| `analyses` | `dict[str, dict[str, polar.DF]]` | Nested: analysis_id → revision_id → dataframe |
| `metadata` | `dict[str, Any]` | Source flight, analyst, timestamps, parameters |

**metadata structure:**
```python
{
    "flight_id": str,                    # Reference to source Flight
    "analysis_type": str,                # e.g., "ppk"
    "created_at": str,                   # ISO 8601
    "analyst": str,                      # Person/system performing analysis
    "source_synchronized_revision": str, # rev_id of input sync data
    "notes": str                         # Free-form notes
}
```

#### Methods

| Signature | Purpose |
|-----------|---------|
| `to_hdf5(filepath: str, compression='gzip') -> None` | Serialize to HDF5 |
| `from_hdf5(filepath: str) -> PostProcessed` | Deserialize from HDF5 |
| `__repr__() -> str` | Show analysis type, revisions, summary |
| `get_revision(analysis_id: str, rev_id: str) -> polar.DF` | Get specific revision |
| `list_revisions(analysis_id: str) -> list[str]` | List available revisions |

#### Constraints

- `analysis_type` must match class subtype
- All DataFrames are `polar.DataFrame`
- Must preserve source metadata for traceability
- Immutable after creation

---

### Container: PPKData (extends PostProcessed)

**Purpose:** Post-Processed Kinematic (PPK) positioning from RTKLIB or similar GNSS post-processing

**Type:** `class` | **Inheritance:** `PostProcessed` | **Temporal:** Multiple revisions per flight

#### Attributes

| Name | Type | Description |
|------|------|-------------|
| `position` | `dict[str, polar.DF]` | Position solutions; keys=revision_ids |
| `statistics` | `dict[str, polar.DF]` | Quality metrics per epoch; keys=revision_ids (match position) |
| `metadata` | `dict[str, dict[str, Any]]` | PPK-specific metadata; keys=revision_ids |

**position[rev_id] columns:**
```
timestamp (datetime64[ns])      - UTC epoch time
latitude (float64)              - WGS84 degrees (-90..90)
longitude (float64)             - WGS84 degrees (-180..180)
altitude (float32)              - Meters above ellipsoid
latitude_std (float32)          - Standard deviation in meters
longitude_std (float32)         - Standard deviation in meters
altitude_std (float32)          - Standard deviation in meters
solution_status (string)        - "FIXED", "FLOAT", "SINGLE", "DGPS", "NO_SOLUTION"
num_satellites (int32)          - Total satellites used
```

**statistics[rev_id] columns:**
```
timestamp (datetime64[ns])      - UTC epoch time
sat (int32)                     - Number of satellites
ratio (float32)                 - Fix success ratio (0..1)
pdop (float32)                  - Position dilution of precision
gdop (float32)                  - Geometric dilution of precision
solution_type (string)          - "FIXED", "FLOAT", "SINGLE"
status (string)                 - "ok", "warning", "error"
```

**metadata[rev_id] structure:**
```python
{
    "revision_id": str,                 # YYYYMMDD_hhmmss
    "created_at": str,                  # ISO 8601
    "rtklib_config": dict,              # Parsed config: pos1-posmode, frequency, troposphere, ionosphere, etc.
    "reference_station": str,           # Base station ID or ECEF coordinates
    "satellite_system": str,            # "GPS+GLONASS+Galileo+BeiDou"
    "troposphere_model": str,           # "Saastamoinen", "SBAS", etc.
    "ionosphere_model": str,            # "Broadcast", "IONEX", "SLANT_TEC"
    "solution_frequency": int,          # Hz (1, 5, 10, etc.)
    "time_range": {
        "start": str,                   # ISO 8601
        "end": str                      # ISO 8601
    },
    "input_rover_file": str,            # Path to rover data used
    "input_base_file": str,             # Path to base station data used
    "notes": str                        # Processing notes
}
```

#### Methods

| Signature | Purpose |
|-----------|---------|
| `to_hdf5(filepath: str, compression='gzip') -> None` | Save to HDF5 with structure: /rev_ID/position/, /rev_ID/statistics/, metadata as attrs |
| `from_hdf5(filepath: str) -> PPKData` | Load from HDF5, reconstruct all revisions |
| `__repr__() -> str` | Show revision count, time ranges, solution breakdown |
| `from_rtklib(solution_pos_path: str, config_path: str, rev_id: str) -> None` | Parse RTKLIB solution.pos and add new revision |
| `compare_revisions(rev1_id: str, rev2_id: str) -> dict` | Stats comparing revisions: num_fixed, num_float, RMS diffs, PDOP |

#### Constraints

- `position` and `statistics` must have matching timestamps and row counts per revision
- All coordinates must be WGS84
- Uncertainty estimates are mandatory (no NaN in _std columns)
- `solution_status` from defined set: "FIXED", "FLOAT", "SINGLE", "DGPS", "NO_SOLUTION"
- Immutable after creation

---

## HDF5 File Hierarchy Schematic

### Flight HDF5 File: `proc/flight_data.h5`

```
flight_data.h5
│
├── metadata (HDF5 attributes)
│   ├── @flight_id = "FLIGHT_001"
│   ├── @timestamp = "2025-02-04T14:30:22Z"
│   ├── @aircraft_id = "DJI_M300_SN123456"
│   ├── @mission_name = "Coastal survey"
│   └── ... (all metadata fields)
│
├── /raw_data/ (group)
│   ├── /raw_data/drone_data/ (group)
│   │   ├── litchi (dataset: [N rows] composite)
│   │   │   ├── @columns = "timestamp;latitude;longitude;altitude;heading;..."
│   │   │   ├── @source = "Litchi CSV export"
│   │   │   └── @timestamp_resolution = "millisecond"
│   │   └── drone (dataset: [N rows] composite)
│   │       ├── @columns = "timestamp;latitude;longitude;altitude;..."
│   │       └── @source = "DJI CSV log"
│   │
│   └── /raw_data/payload/ (group)
│       ├── gps (dataset: [N rows] composite)
│       │   ├── @columns = "timestamp;latitude;longitude;altitude;hdop;vdop;num_sat"
│       │   └── @source = "Reach RS2 GNSS receiver"
│       ├── /raw_data/payload/inclinometer/ (group)
│       │   ├── imu_1 (dataset: [N rows, 9 cols])
│       │   │   ├── @columns = "timestamp;roll;pitch;yaw;acc_x;acc_y;acc_z;gyro_x;gyro_y;gyro_z"
│       │   │   └── @source = "Xsens MTi-630 IMU #1"
│       │   └── imu_2 (dataset: [N rows, 9 cols])
│       ├── adc (dataset: [N rows] composite)
│       │   ├── @columns = "timestamp;channel_0;channel_1;channel_2;channel_3"
│       │   └── @source = "NI DAQ ADC"
│       └── /raw_data/payload/sensors/ (group)
│           ├── temperature (dataset)
│           └── pressure (dataset)
│
└── /synchronized_data/ (group)
    ├── rev_20250204_143022 (dataset: [N rows, M cols] composite)
    │   ├── @revision_id = "20250204_143022"
    │   ├── @created_at = "2025-02-04T14:32:15Z"
    │   ├── @synchronization_method = "cubic_spline"
    │   ├── @reference_clock = "gps"
    │   ├── @compression = "gzip (level 4)"
    │   ├── @columns = "timestamp;drone_lat;drone_lon;drone_alt;gps_lat;gps_lon;imu_roll;imu_pitch;imu_yaw;adc_channel_0;..."
    │   └── ... (all metadata attributes)
    │
    └── rev_20250204_150015 (dataset: [N rows, M cols] composite)
        ├── @revision_id = "20250204_150015"
        ├── @synchronization_method = "linear_interpolation"
        ├── @reference_clock = "imu"
        └── ... (metadata)
```

### PPK HDF5 File: `proc/ppk/ppk_solution.h5`

```
ppk_solution.h5
│
├── metadata (HDF5 attributes)
│   ├── @flight_id = "FLIGHT_001"
│   ├── @analysis_type = "ppk"
│   ├── @created_at = "2025-02-04T16:00:00Z"
│   └── @analyst = "GNSS processor v2.1"
│
└── /ppk/ (group)
    ├── rev_20250204_143022/ (group)
    │   ├── position (dataset: [N rows, 8 cols])
    │   │   ├── @columns = "timestamp;latitude;longitude;altitude;latitude_std;longitude_std;altitude_std;solution_status"
    │   │   ├── @compression = "gzip (level 4)"
    │   │   └── @dtype = "composite"
    │   │
    │   ├── statistics (dataset: [N rows, 7 cols])
    │   │   ├── @columns = "timestamp;sat;ratio;pdop;gdop;solution_type;status"
    │   │   └── @compression = "gzip (level 4)"
    │   │
    │   └── metadata (HDF5 attributes on group)
    │       ├── @revision_id = "20250204_143022"
    │       ├── @created_at = "2025-02-04T14:32:15Z"
    │       ├── @rtklib_config = "JSON: {\"pos1-posmode\": \"kinematic\", ...}"
    │       └── ... (all metadata)
    │
    └── rev_20250204_150015/ (group)
        ├── position (dataset: [N rows, 8 cols])
        ├── statistics (dataset: [N rows, 7 cols])
        └── metadata (attributes)
```

---

## Filesystem Layout Schematic

```
flight_path/
│
├── aux/                                    # Auxiliary raw data (original from flight)
│   ├── 20250204_143022_config.yml
│   ├── 20250204_143022_file.log
│   ├── camera/
│   │   ├── 20250204_143022_video.mp4
│   │   └── 20250204_143022_video.xml
│   └── sensors/
│       ├── 20250204_143022_ACC.bin
│       ├── 20250204_143022_ADC.bin
│       ├── 20250204_143022_*.bin (various sensor binary)
│       ├── 20250204_143022_*.csv (sensor CSV exports)
│       └── 20250204_143022_*.log (sensor logs)
│
├── drone/                                  # Drone manufacturer data
│   ├── 20250204_143022_drone.csv          # DJI CSV log
│   ├── 20250204_143022_drone.dat          # DJI binary log
│   └── 20250204_143022_litchi.csv         # Litchi export
│
└── proc/                                   # All processed
    ├── flight_data.h5                      # ★ MAIN: All raw_data + all sync revisions
    │
    ├── ppk/                                # PPK analysis
    │   ├── ppk_solution.h5                 # ★ ALL PPK REVISIONS consolidated
    │   ├── rev_20250204_143022/            # Source files for revision 1
    │   │   ├── rtklib.conf
    │   │   ├── solution.pos
    │   │   └── solution.pos.stat
    │   └── rev_20250204_150015/            # Source files for revision 2
    │       ├── rtklib.conf
    │       ├── solution.pos
    │       └── solution.pos.stat
    │
    └── [other_analyses]/                   # Other analysis types
        ├── analysis_results.h5
        └── rev_*/
```

---

## HDF5 Dataset Specifications

### Format for All Datasets

```
Path: /group/subgroup/dataset_name
DataType: float32 | float64 | int32 | int64 | string | composite
Shape: [N rows] or [N rows, M cols]
Chunks: [10000, auto] for time-series (enables efficient subset reads)
Compression: gzip (level 4) | lzf | none

Attributes (HDF5 key-value pairs):
  @columns: "col1;col2;col3" (semicolon-separated column names, or JSON for complex)
  @dtype: "composite" or "float32" etc.
  @source: "Description of data source or processing"
  @units: "degrees;degrees;meters" (matching column order, or JSON)
  @timestamp_format: "ISO 8601" or custom
  @timestamp_resolution: "millisecond", "10ms", "100ms", "1s"
  @compression: "gzip (level 4)"
  @[any other relevant metadata]
```

### Flight Raw Data Datasets

```
Path: /raw_data/drone_data/litchi
DataType: composite (multiple columns)
Shape: [3625 rows, N columns]
Compression: gzip (level 4)
Attributes:
  @columns = "timestamp;latitude;longitude;altitude;heading;speed_x;speed_y;speed_z;battery_percent"
  @source = "Litchi CSV flight plan export"
  @units = "seconds UTC;degrees;degrees;meters;degrees;m/s;m/s;m/s;percent"
  @timestamp_resolution = "millisecond"
  @dtype = "composite"

Path: /raw_data/payload/gps
DataType: composite
Shape: [7250 rows, 8 columns]
Compression: gzip (level 4)
Attributes:
  @columns = "timestamp;latitude;longitude;altitude;hdop;vdop;num_satellites;fix_type"
  @source = "Reach RS2 GNSS receiver"
  @units = "seconds UTC;degrees;degrees;meters;unitless;unitless;count;enum"
  @timestamp_resolution = "500ms"
  @coordinates = "WGS84"
```

### Synchronized Data Dataset

```
Path: /synchronized_data/rev_20250204_143022
DataType: composite (all aligned columns)
Shape: [1824 rows, 27 columns]
Chunks: [10000, auto]
Compression: gzip (level 4)

Attributes:
  @revision_id = "20250204_143022"
  @created_at = "2025-02-04T14:32:15Z"
  @columns = "timestamp;drone_lat;drone_lon;drone_alt;...;gps_lat;gps_lon;...;imu_roll;imu_pitch;..."
  @synchronization_method = "cubic_spline"
  @reference_clock = "gps"
  @interpolation_method = "cubic"
  @timestamp_tolerance_ms = 100.0
  @num_records = 1824
  @time_range_start = "2025-02-04T14:30:22Z"
  @time_range_end = "2025-02-04T14:31:29Z"
  @sources_used = "drone_data;gps;inclinometer;adc"
  @units = "seconds UTC;degrees;degrees;meters;...;degrees;degrees;...;radians;radians;..."
  @notes = "Initial sync, all sensors resampled to 1 Hz"
```

### PPK Position Dataset

```
Path: /ppk/rev_20250204_143022/position
DataType: composite
Shape: [1824 rows, 8 columns]
Chunks: [10000, auto]
Compression: gzip (level 4)

Attributes:
  @columns = "timestamp;latitude;longitude;altitude;latitude_std;longitude_std;altitude_std;solution_status"
  @source = "RTKLIB solution.pos"
  @units = "seconds UTC;degrees;degrees;meters;meters;meters;meters;enum"
  @coordinates = "WGS84"
  @solution_types = "FIXED|FLOAT|SINGLE|DGPS|NO_SOLUTION"
  @dtype = "composite"
```

### PPK Statistics Dataset

```
Path: /ppk/rev_20250204_143022/statistics
DataType: composite
Shape: [1824 rows, 7 columns]
Chunks: [10000, auto]
Compression: gzip (level 4)

Attributes:
  @columns = "timestamp;sat;ratio;pdop;gdop;solution_type;status"
  @source = "RTKLIB solution.pos.stat"
  @units = "seconds UTC;count;ratio;unitless;unitless;enum;enum"
  @solution_types = "FIXED|FLOAT|SINGLE"
  @dtype = "composite"
```

---

## Serialization Strategy

### Flight.to_hdf5()

**Algorithm:**
1. Create HDF5 file
2. Write root metadata dict as HDF5 attributes (flight_id, timestamp, aircraft_id, etc.)
3. Create `/raw_data/` group
   - Create `/raw_data/drone_data/` group
     - For each drone source (litchi, dji): convert polar.DataFrame to HDF5 dataset
     - Write columns, source, units as attributes
   - Create `/raw_data/payload/` group
     - For each sensor (gps, inclinometer, adc, sensors): write as nested groups/datasets
     - Write all attributes (columns, source, units, timestamp_resolution)
4. Create `/synchronized_data/` group
   - For each revision_id in synchronized_data dict:
     - Write DataFrame as dataset
     - Write revision metadata as HDF5 attributes (revision_id, created_at, method, etc.)
5. Set compression on all datasets: gzip level 4
6. Set chunking: [10000, auto] for time-series
7. Flush and close file

**Pseudo-code:**
```python
def to_hdf5(self, filepath: str, compression: str = 'gzip'):
    with h5py.File(filepath, 'w') as f:
        # Root attributes
        for key, val in self.metadata.items():
            f.attrs[key] = val
        
        # Raw data
        raw_group = f.create_group('raw_data')
        drone_group = raw_group.create_group('drone_data')
        for source_name, df in self.raw_data.drone_data.items():
            ds = drone_group.create_dataset(source_name, data=df_to_h5(df),
                                           compression=compression, chunks=(10000, None))
            ds.attrs['columns'] = ';'.join(df.columns)
            ds.attrs['source'] = f"Drone {source_name}"
            # ... add more attributes
        
        payload_group = raw_group.create_group('payload')
        # ... similar for GPS, IMU, ADC, etc.
        
        # Synchronized data
        sync_group = f.create_group('synchronized_data')
        for rev_id, df in self.synchronized_data.items():
            ds = sync_group.create_dataset(rev_id, data=df_to_h5(df),
                                          compression=compression, chunks=(10000, None))
            for key, val in self.synchronized_data_metadata[rev_id].items():
                ds.attrs[key] = val
```

### Flight.from_hdf5()

**Algorithm:**
1. Open HDF5 file
2. Read root attributes → metadata dict
3. Read `/raw_data/drone_data/` datasets → populate raw_data.drone_data dict
4. Read `/raw_data/payload/` groups/datasets → populate raw_data.payload dict
5. Read `/synchronized_data/` datasets
   - For each revision_id:
     - Read DataFrame
     - Read metadata attributes
     - Validate: timestamp strictly monotonic, coordinates in valid ranges
     - Add to synchronized_data dict
6. Validate entire Flight (all required fields present)
7. Return Flight instance
8. Close file

**Pseudo-code:**
```python
@classmethod
def from_hdf5(cls, filepath: str) -> Flight:
    with h5py.File(filepath, 'r') as f:
        # Metadata
        metadata = dict(f.attrs)
        
        # Raw data
        raw_data = RawData()
        for source in f['/raw_data/drone_data']:
            raw_data.drone_data[source] = h5_to_df(f[f'/raw_data/drone_data/{source}'])
        
        for sensor in f['/raw_data/payload']:
            # Handle nested structure (inclinometer has multiple sensors)
            raw_data.payload[sensor] = h5_to_df(f[f'/raw_data/payload/{sensor}'])
        
        # Synchronized data
        synchronized_data = {}
        for rev_id in f['/synchronized_data']:
            ds = f[f'/synchronized_data/{rev_id}']
            df = h5_to_df(ds)
            # Validate
            assert df['timestamp'].is_strictly_monotonic()
            sync_meta = dict(ds.attrs)
            # Note: metadata stored separately or reconstructed
            synchronized_data[rev_id] = df
        
        return cls(metadata=metadata, raw_data=raw_data, synchronized_data=synchronized_data)
```

### PPKData.to_hdf5()

**Algorithm:**
1. Create HDF5 file
2. Write root metadata as attributes
3. Create `/ppk/` group
4. For each revision_id in position dict:
   - Create `/ppk/{revision_id}/` group
   - Write position[revision_id] as dataset (with attributes)
   - Write statistics[revision_id] as dataset (with attributes)
   - Write metadata[revision_id] dict as group attributes
5. Set compression: gzip level 4
6. Close file

**Pseudo-code:**
```python
def to_hdf5(self, filepath: str, compression: str = 'gzip'):
    with h5py.File(filepath, 'w') as f:
        # Root metadata
        for key, val in self.metadata.items():
            f.attrs[key] = val
        
        # PPK revisions
        ppk_group = f.create_group('ppk')
        for rev_id in self.position.keys():
            rev_group = ppk_group.create_group(rev_id)
            
            # Position dataset
            pos_ds = rev_group.create_dataset('position', 
                                             data=df_to_h5(self.position[rev_id]),
                                             compression=compression, chunks=(10000, None))
            pos_ds.attrs['columns'] = ';'.join(self.position[rev_id].columns)
            # ... add more attributes
            
            # Statistics dataset
            stat_ds = rev_group.create_dataset('statistics',
                                              data=df_to_h5(self.statistics[rev_id]),
                                              compression=compression, chunks=(10000, None))
            # ... add attributes
            
            # Metadata as group attributes
            for key, val in self.metadata[rev_id].items():
                if isinstance(val, dict):
                    rev_group.attrs[key] = json.dumps(val)
                else:
                    rev_group.attrs[key] = val
```

### PPKData.from_hdf5()

**Algorithm:**
1. Open HDF5 file
2. Read root attributes
3. Iterate `/ppk/{rev_id}/` groups:
   - Read position dataset
   - Read statistics dataset
   - Read group attributes → metadata[rev_id]
   - Validate: matching timestamps, solution_status values, coordinates
4. Return PPKData instance

---

## Versioning & Revision Management

### Revision ID Format

**Standard:** `YYYYMMDD_hhmmss` (14 chars, UTC)
- Example: `20250204_143022`

**With version counter:** `YYYYMMDD_hhmmss_VN` (if multiple in same second)
- Example: `20250204_143022_01`, `20250204_143022_02`

### Immutable Versioning Principle

**Core rule:** Once created, a revision never changes. New processing creates new revision.

**Workflow example (synchronization):**
```
1. flight_data.h5 has: /synchronized_data/rev_20250204_143022
2. Analyst performs new sync with different parameters
3. Create new SynchronizedData instance (doesn't touch old)
4. Add to Flight.synchronized_data dict: rev_20250204_150015
5. Call Flight.to_hdf5()
6. Result: file now has BOTH old and new revisions
7. User can compare or choose which to use next
```

**Workflow example (PPK):**
```
1. ppk_solution.h5 has: /ppk/rev_20250204_143022
2. Analyst runs PPK with new RTKLIB config
3. Parse solution.pos, create new DataFrames
4. Create new revision group: /ppk/rev_20250204_150015
5. Result: file consolidates both, user calls compare_revisions()
```

### Metadata Tracking

Every revision includes complete metadata for full reproducibility:

**Synchronized data metadata captures:**
- `revision_id`: Unique identifier
- `created_at`: ISO 8601 timestamp of sync
- `synchronization_method`: Algorithm (cubic_spline, linear, etc.)
- `reference_clock`: Primary sensor (gps, imu, drone)
- `interpolation_method`: Upsampling/downsampling approach
- `timestamp_tolerance_ms`: Acceptable drift
- `sources_used`: Which sensors included

**PPK metadata captures:**
- `revision_id`: Unique identifier
- `created_at`: ISO 8601 timestamp
- `rtklib_config`: Complete configuration dict (reproducible)
- `reference_station`: Base station used
- `satellite_system`: GPS/GLONASS/Galileo/BeiDou combo
- `troposphere_model`, `ionosphere_model`: Processing choices
- `input_rover_file`, `input_base_file`: Input data references

This enables full reproducibility, audit trails, and comparison.

---

## Data Types & Precision

| Data Category | Type | Precision | Rationale |
|---------------|------|-----------|-----------|
| **Temporal** | `datetime64[ns]` | Nanosecond | UTC, high precision for fusion, HDF5 native |
| **Latitude/Longitude** | `float64` | ~1cm @ equator | Geographic calculations need precision |
| **Altitude** | `float32` | ~0.1mm | Sufficient, 4x less memory |
| **Uncertainty (std)** | `float32` | ~0.1mm | Stdev doesn't need double precision |
| **Velocity** | `float32` | ~1e-6 m/s | Sufficient for drones |
| **Acceleration** | `float32` | ~1e-6 m/s² | Sufficient for IMU |
| **Angular (radians)** | `float32` | ~1e-7 rad (~6 arcsec) | Sufficient for attitude |
| **Counts/Satellites** | `int32` | Exact | Never negative, max < 100 |
| **Indices** | `int64` | Exact | For large datasets |
| **Status/Type** | `string` | Categorical | FIXED, FLOAT, SINGLE, DGPS, etc. |
| **Binary** | `uint8` or `bytes` | Exact | Raw ADC, camera data |

---

## Validation Rules

### On Load (from_HDF5)

| Rule | Applies To | Error Handling |
|------|-----------|-----------------|
| Timestamp strictly monotonic increasing | All DataFrames with timestamp | Raise ValueError with indices |
| Latitude in [-90, 90] degrees | GPS, drone position | Raise ValueError, suggest swap |
| Longitude in [-180, 180] degrees | GPS, drone position | Raise ValueError, suggest wrap |
| Altitude < 50 km | Drone altitude | Warn, provide statistics |
| Solution status in defined set | PPKData solution_status | Warn about unknowns, don't reject |
| Position and statistics timestamps match | PPKData per revision | Raise ValueError if mismatch |
| Required attributes present | All HDF5 groups/datasets | Raise ValueError listing missing |
| No NaN in key columns | timestamp, lat, lon (position data) | Raise ValueError with count |
| PDOP/GDOP in reasonable range | PPK statistics | Warn if > 100 (highly degraded) |

### On Creation (to_HDF5)

| Rule | Applies To | Error Handling |
|------|-----------|-----------------|
| Metadata contains required fields | Flight, PPKData | Raise ValueError |
| Revision ID matches format | All revisions | Raise ValueError if invalid |
| No duplicate revision IDs | synchronized_data, PPK | Raise ValueError if adding duplicate |
| Position and statistics same length | PPKData per revision | Raise ValueError if mismatch |
| All DataFrames are polar.DataFrame | Flight, PostProcessed | Raise TypeError if pandas |

---

## Usage Examples

### Example 1: Load Flight and Access Synchronized Data

```python
# Load complete flight
flight = Flight.from_hdf5('proc/flight_data.h5')
print(flight)
# Output: Flight(flight_id='FLIGHT_001', aircraft='DJI_M300_SN123456', raw_data=125MB, sync_versions=2)

# Access raw drone data
raw_drone = flight.raw_data.drone_data['litchi']
print(raw_drone.columns)  # [timestamp, latitude, longitude, altitude, ...]

# List synchronized revisions
revisions = flight.list_synchronized_revisions()
# Returns: ['20250204_143022', '20250204_150015']

# Get specific synchronized revision
sync_v1 = flight.get_synchronized_revision('20250204_143022')
print(sync_v1.metadata['synchronization_method'])  # "cubic_spline"
print(sync_v1.metadata['reference_clock'])  # "gps"
```

### Example 2: Load and Compare PPK Solutions

```python
# Load PPK
ppk = PPKData.from_hdf5('proc/ppk/ppk_solution.h5')
print(ppk)
# Output: PPKData(flight_id='FLIGHT_001', revisions=2)

# List revisions
revisions = ppk.list_revisions('ppk')
# Returns: ['20250204_143022', '20250204_150015']

# Get position from first revision
pos_v1 = ppk.get_revision('position', '20250204_143022')
print(pos_v1.columns)
# [timestamp, latitude, longitude, altitude, latitude_std, longitude_std, altitude_std, solution_status]

# Count fixed solutions
num_fixed = pos_v1.filter(pl.col('solution_status') == 'FIXED').shape[0]

# Compare revisions
comparison = ppk.compare_revisions('20250204_143022', '20250204_150015')
# Returns: {
#   'num_fixed_v1': 1512, 'num_fixed_v2': 1724,
#   'num_float_v1': 252, 'num_float_v2': 92,
#   'mean_pdop_v1': 3.2, 'mean_pdop_v2': 2.1,
#   'rms_horiz_diff': 0.045, 'rms_vert_diff': 0.032
# }
```

### Example 3: Create New Synchronized Revision

```python
# Load existing flight
flight = Flight.from_hdf5('proc/flight_data.h5')

# Perform new synchronization
new_sync_df = perform_synchronization(
    flight.raw_data,
    method='linear_interpolation',
    reference_clock='imu',
    target_frequency=10
)

# Create SynchronizedData
new_sync = SynchronizedData(
    data=new_sync_df,
    metadata={
        'revision_id': '20250204_161530',
        'created_at': '2025-02-04T16:15:30Z',
        'synchronization_method': 'linear_interpolation',
        'reference_clock': 'imu',
        'interpolation_method': 'linear',
        'timestamp_tolerance_ms': 50.0,
        'num_records': len(new_sync_df),
        'time_range': {
            'start': new_sync_df['timestamp'].min().isoformat(),
            'end': new_sync_df['timestamp'].max().isoformat()
        },
        'sources_used': ['drone_data', 'gps', 'inclinometer', 'adc'],
        'notes': 'Revised sync with IMU primary clock'
    }
)

# Add and persist
flight.synchronized_data['20250204_161530'] = new_sync
flight.to_hdf5('proc/flight_data.h5', compression='gzip')
# File now has: v1, v2, v3
```

### Example 4: Create New PPK Revision

```python
# Load existing PPK
ppk = PPKData.from_hdf5('proc/ppk/ppk_solution.h5')

# Parse new RTKLIB solution
from pathlib import Path
rtklib_dir = Path('proc/ppk/rev_20250204_165000')
ppk.from_rtklib(
    solution_pos_path=rtklib_dir / 'solution.pos',
    config_path=rtklib_dir / 'rtklib.conf',
    revision_id='20250204_165000'
)

# Check improvement
comparison = ppk.compare_revisions('20250204_143022', '20250204_165000')
print(f"Fixed: {comparison['num_fixed_v1']} → {comparison['num_fixed_v2']}")
print(f"PDOP: {comparison['mean_pdop_v1']:.2f} → {comparison['mean_pdop_v2']:.2f}")

# Persist
ppk.to_hdf5('proc/ppk/ppk_solution.h5', compression='gzip')
```

---

## Notes & Edge Cases

### Data Precision

- **WGS84 Coordinates:** ALWAYS float64 (degrees). Never float32 for lat/lon. Preserves ~1cm precision.
- **Altitude:** float32 sufficient (~0.1mm). 4x less memory than float64.
- **Uncertainties:** float32 appropriate (stdev doesn't need double precision).
- **Timestamps:** datetime64[ns] avoids floating-point precision loss.

### Large Datasets

- **Chunking:** [10000, auto] for time-series enables efficient subset reads.
- **Compression:** gzip level 4 (good balance: smaller files, fast decode).
- **Lazy Loading:** HDF5 supports reading subsets without loading entire file.
- **Memory:** polar.DataFrame more efficient than pandas for large datasets.

### Temporal Alignment

- **Different Resolutions:** Raw sensors may have 100ms, 1s, 10s timestamps. Sync resamples to common grid.
- **Clock Drift:** Small timing offsets between sources require interpolation tolerance.
- **Gaps:** If sensor fails mid-flight, synchronized data has NaN or rows dropped (design choice).

### Reproducibility

- **Never modify past revisions:** New analysis = new revision entry, not edit.
- **Document configuration:** Store full config files (rtklib.conf, processing params) in metadata.
- **Track lineage:** metadata indicates what input versions were used.
- **Preserve raw files:** Keep aux/, drone/ folders for audit trail.
- **Version software:** Include processing software version in metadata.

### Coordinate Systems

- **All positions must be WGS84** (lat/lon degrees) or consistently documented.
- Never mix coordinate systems without explicit conversion.
- Altitudes always "above ellipsoid" unless clearly stated.
- Local NED (North-East-Down) for relative positions stored separately.

### Multi-Base-Station PPK

- Each base station config = separate revision (different results).
- Use PPKData.compare_revisions() to understand impact.
- Metadata['reference_station'] documents which was used.

---

## Implementation Recommendations

1. **Start with Flight:** Implement raw_data and metadata first, defer synchronized_data.
2. **Use Polar DataFrames:** More memory-efficient than pandas for large datasets.
3. **Validate incrementally:** Start with basic dtype checks, add constraints over time.
4. **Test round-trip:** Verify to_hdf5 → from_hdf5 preserves all data and metadata.
5. **Plan versioning early:** Design revision_id format before first release.
6. **Document coordinate systems:** Be explicit about WGS84 vs local vs NED.
7. **Keep raw files:** Don't delete aux/, drone/ after consolidating to HDF5.
8. **Version schema:** Include schema version in root HDF5 attributes for future compatibility.
9. **Chunking strategy:** Use [10000, auto] for time-series, default for other data.
10. **Compression:** Start with gzip level 4 (good speed/size balance).