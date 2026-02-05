# Drone Platforms

PILS supports multiple drone platforms with automatic detection.

## Supported Platforms

| Platform | Manufacturer | File Formats | Detection |
|----------|--------------|--------------|-----------|
| DJI | DJI | `.csv`, `.DAT` | CSV header pattern |
| BlackSquare | ArduPilot | `.BIN`, `.bin`, `.log` | Binary header 0xA395 |
| Litchi | Litchi | `.csv` | Column names |

## Auto-Detection

When you call `flight.add_drone_data()`, PILS automatically:

1. Scans the `drone_data/` directory
2. Identifies file types
3. Loads the appropriate parser
4. Populates `flight.drone_data`

```python
flight = Flight(flight_info)
flight.add_drone_data()

# Check detected platform
print(f"Keys: {list(flight.drone_data.keys())}")
```

---

## DJI Drones

Support for Phantom, Mavic, Matrice, and other DJI platforms.

### Supported Formats

=== "CSV Export"

    DATCON CSV exports:
    
    ```
    datetime,latitude,longitude,altitude,height,velocity_x,velocity_y,
    velocity_z,pitch,roll,yaw,gimbal_pitch,gimbal_roll,gimbal_yaw,...
    ```

=== "DAT Files"

    Encrypted binary flight logs:
    
    - `.DAT` files from DJI Matrice 600
    - Internal decryption tool available
    - XOR-encrypted messages
    - Organized by message type (GPS, RTK)

??? info "DAT File Format & Decoding"

    DAT files contain encrypted binary messages that are parsed and decrypted by PILS.

    #### Message Structure

    Each message in a DAT file follows this structure:

    ```
    Byte 0:       0x55 (marker)
    Byte 1:       Message length
    Bytes 2-4:    Reserved
    Bytes 4-5:    Message type (little-endian uint16)
    Byte 6:       XOR encryption key
    Bytes 6-9:    Tick (timestamp, little-endian uint32)
    Bytes 10+:    Encrypted payload
    ```

    #### Decoding Process

    1. **Split messages**: File is split on `0x55` marker bytes
    2. **Extract header**: Parse message type, encryption key, and tick
    3. **Decrypt payload**: XOR each payload byte with the key byte
    4. **Unpack fields**: Use struct format codes to extract fields from specific byte offsets
    5. **Convert units**: Apply conversion functions (e.g., lat/lon scaling)
    6. **Organize by type**: Group messages into separate DataFrames by type

    #### Supported Message Types

    | Message Type | ID | Description | Payload Size |
    |--------------|-----|-------------|--------------|
    | GPS | 2096 | Standard GPS data | 66 bytes |
    | RTK | 53234 | RTK positioning data | 72 bytes |

    #### GPS Message (Type 2096)

    | Field | Format | Offset | Unit | Conversion | Description |
    |-------|--------|--------|------|------------|-------------|
    | `date` | `I` | 0 | - | - | Date (YYYYMMDD) |
    | `time` | `I` | 4 | - | - | Time (HHMMSS) |
    | `longitude` | `i` | 8 | degrees | ÷ 1e7 | GPS longitude |
    | `latitude` | `i` | 12 | degrees | ÷ 1e7 | GPS latitude |
    | `heightMSL` | `i` | 16 | m | ÷ 1000 | Height above MSL |
    | `velN` | `f` | 20 | m/s | ÷ 100 | North velocity |
    | `velE` | `f` | 24 | m/s | ÷ 100 | East velocity |
    | `velD` | `f` | 28 | m/s | ÷ 100 | Down velocity |
    | `hdop` | `f` | 32 | - | - | Horizontal DOP |
    | `pdop` | `f` | 36 | - | - | Position DOP |
    | `hacc` | `f` | 40 | m | - | Horizontal accuracy |
    | `sacc` | `f` | 44 | m | - | Speed accuracy |
    | `numGPS` | `I` | 56 | - | - | GPS satellites |
    | `numGLN` | `I` | 60 | - | - | GLONASS satellites |
    | `numSV` | `H` | 64 | - | - | Total satellites |

    #### RTK Message (Type 53234)

    | Field | Format | Offset | Unit | Conversion | Description |
    |-------|--------|--------|------|------------|-------------|
    | `date` | `I` | 0 | - | - | Date (YYYYMMDD) |
    | `time` | `I` | 4 | - | - | Time (HHMMSS) |
    | `lon_p` | `d` | 8 | degrees | - | RTK longitude (primary) |
    | `lat_p` | `d` | 16 | degrees | - | RTK latitude (primary) |
    | `hmsl_p` | `f` | 24 | m | - | Height MSL (primary) |
    | `lon_s` | `i` | 28 | degrees | ÷ 1e7 | Longitude (secondary) |
    | `lat_s` | `i` | 32 | degrees | ÷ 1e7 | Latitude (secondary) |
    | `hmsl_s` | `i` | 36 | m | ÷ 1000 | Height MSL (secondary) |
    | `vel_n` | `f` | 40 | m/s | - | North velocity |
    | `vel_e` | `f` | 44 | m/s | - | East velocity |
    | `vel_d` | `f` | 48 | m/s | - | Down velocity |
    | `yaw` | `h` | 50 | degrees | - | Yaw angle |
    | `svn_s` | `B` | 52 | - | - | Satellites (secondary) |
    | `svn_p` | `B` | 53 | - | - | Satellites (primary) |
    | `hdop` | `f` | 54 | - | - | Horizontal DOP |
    | `pitch` | `f` | 58 | degrees | - | Pitch angle |
    | `pos_flg_0-5` | `B` | 62-67 | - | - | Position flags |
    | `gps_state` | `H` | 68 | - | - | GPS state code |

    !!! note "Struct Format Codes"
        - `I` = unsigned int (4 bytes)
        - `i` = signed int (4 bytes)
        - `f` = float (4 bytes)
        - `d` = double (8 bytes)
        - `H` = unsigned short (2 bytes)
        - `h` = signed short (2 bytes)
        - `B` = unsigned byte (1 byte)

    #### Loading DAT Files

    ```python
    from pils.drones import DJIDrone

    # Load DAT file
    dji = DJIDrone(path="/path/to/flight.DAT")
    dji.load_data(use_dat=True)

    # Access message types
    gps_df = dji.data['GPS']
    rtk_df = dji.data['RTK']

    print(f"GPS records: {gps_df.height}")
    print(f"RTK records: {rtk_df.height}")

    # Check columns
    print(f"GPS columns: {gps_df.columns}")
    ```

    #### Example: DAT File Analysis

    ```python
    import polars as pl

    # Load DAT file
    dji = DJIDrone("/path/to/flight.DAT")
    dji.load_data(use_dat=True)

    # Analyze GPS data
    gps = dji.data['GPS']
    print(f"GPS samples: {gps.height}")
    print(f"Time range: {gps['tick'].min()} - {gps['tick'].max()} ticks")

    # Check satellite count
    sat_stats = gps.select([
        pl.col('GPS:numSV').mean().alias('avg_satellites'),
        pl.col('GPS:numGPS').mean().alias('avg_gps_sats'),
        pl.col('GPS:numGLN').mean().alias('avg_glonass_sats'),
    ])
    print(sat_stats)

    # Check position accuracy
    print(f"Horizontal accuracy: {gps['GPS:hacc'].mean():.2f} m")

    # Analyze RTK data (if available)
    if 'RTK' in dji.data:
        rtk = dji.data['RTK']
        print(f"RTK samples: {rtk.height}")
        print(f"RTK HDOP: {rtk['RTK:hdop'].mean():.2f}")
    ```

    print(f"RTK HDOP: {rtk['RTK:hdop'].mean():.2f}")
```

### Loading

```python
from pils.drones import DJIDrone

# Via flight container
flight.add_drone_data()
dji_data = flight.drone_data['data']

# Or directly
dji = DJIDrone(path="/path/to/flight.csv")
dji.load_data()
data = dji.data
```

### Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `tick` | `Int64` | - | Internal tick counter |
| `datetime` | `Datetime` | - | UTC timestamp |
| `latitude` | `Float64` | degrees | GPS latitude |
| `longitude` | `Float64` | degrees | GPS longitude |
| `altitude_gps` | `Float64` | m | GPS altitude (MSL) |
| `altitude_barometric` | `Float64` | m | Barometric altitude |
| `height_above_takeoff` | `Float64` | m | Relative altitude |
| `velocity_x` | `Float64` | m/s | Forward velocity (body) |
| `velocity_y` | `Float64` | m/s | Right velocity (body) |
| `velocity_z` | `Float64` | m/s | Down velocity (body) |
| `velocity_h` | `Float64` | m/s | Horizontal speed |
| `pitch` | `Float64` | degrees | Aircraft pitch |
| `roll` | `Float64` | degrees | Aircraft roll |
| `yaw` | `Float64` | degrees | Aircraft yaw |
| `gimbal_pitch` | `Float64` | degrees | Gimbal pitch |
| `gimbal_roll` | `Float64` | degrees | Gimbal roll |
| `gimbal_yaw` | `Float64` | degrees | Gimbal yaw |
| `battery_percent` | `Int32` | % | Battery level |
| `flight_mode` | `Utf8` | - | P-GPS, ATTI, etc. |

### Methods

```python
from pils.drones import DJIDrone

dji = DJIDrone(path="/path/to/flight.csv")
dji.load_data()

# Get formatted datetime
formatted = dji.format_date_time()

# Get flight duration
duration = dji.get_flight_duration()
print(f"Duration: {duration} seconds")

# Get bounding box
bounds = dji.get_bounds()
print(f"Lat range: {bounds['lat_min']:.6f} to {bounds['lat_max']:.6f}")
```

### Example Analysis

```python
import polars as pl

dji_df = flight.drone_data['data']

# Flight statistics
stats = dji_df.select([
    pl.col('altitude_gps').max().alias('max_altitude'),
    pl.col('velocity_h').max().alias('max_speed'),
    pl.col('battery_percent').min().alias('min_battery'),
])
print(stats)

# Detect takeoff/landing
takeoff = dji_df.filter(pl.col('height_above_takeoff') > 1).head(1)
landing = dji_df.filter(pl.col('height_above_takeoff') > 1).tail(1)
```

---

## BlackSquare (ArduPilot)

Support for BlackSquare and other ArduPilot-based drones.

### Supported Formats

- `.BIN` - Binary DataFlash logs
- `.bin` - Same as BIN
- `.log` - MAVLink telemetry logs

### Message Types

ArduPilot logs contain multiple message types, each in a separate DataFrame:

| Message | Description | Key Fields |
|---------|-------------|------------|
| `GPS` | GPS data | Lat, Lng, Alt, Spd, Status |
| `IMU` | IMU data | GyrX/Y/Z, AccX/Y/Z |
| `BARO` | Barometer | Alt, Press, Temp |
| `ATT` | Attitude | Roll, Pitch, Yaw |
| `MAG` | Magnetometer | MagX/Y/Z |
| `BAT` | Battery | Volt, Curr, RemPct |
| `RCOU` | RC outputs | Ch1-Ch14 |
| `POS` | Position | Lat, Lng, Alt |
| `PARM` | Parameters | Name, Value |
| `MODE` | Flight mode | Mode, ModeNum |
| `MSG` | Messages | Message text |
| `ERR` | Errors | Subsys, ECode |
| `GPA` | GPS accuracy | VDop, HAcc, VAcc, SAcc |

### Loading

```python
from pils.drones import BlackSquareDrone

# Via flight container
flight.add_drone_data()

# Access specific messages
gps_df = flight.drone_data['GPS']
imu_df = flight.drone_data['IMU']
att_df = flight.drone_data['ATT']

# Or directly
bs = BlackSquareDrone(path="/path/to/flight.BIN")
bs.load_data()
data = bs.data  # Dict[str, pl.DataFrame]
```

### GPS Message Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Time since boot |
| `Status` | `Int32` | - | 0-5, fix status |
| `GMS` | `Int64` | ms | GPS milliseconds of week |
| `GWk` | `Int32` | - | GPS week number |
| `NSats` | `Int32` | - | Satellites used |
| `HDop` | `Float64` | - | HDOP × 100 |
| `Lat` | `Float64` | - | Latitude × 1e7 |
| `Lng` | `Float64` | - | Longitude × 1e7 |
| `Alt` | `Float64` | - | Altitude × 1000 (mm) |
| `Spd` | `Float64` | - | Speed × 100 (cm/s) |
| `GCrs` | `Float64` | - | Course × 100 |
| `VZ` | `Float64` | - | Vertical velocity × 100 |

!!! warning "Scaling"
    ArduPilot uses integer scaling. Convert:
    ```python
    gps_df = gps_df.with_columns([
        (pl.col('Lat') / 1e7).alias('latitude'),
        (pl.col('Lng') / 1e7).alias('longitude'),
        (pl.col('Alt') / 1000).alias('altitude_m'),
        (pl.col('Spd') / 100).alias('speed_ms'),
    ])
    ```

### IMU Message Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Time since boot |
| `GyrX` | `Float64` | rad/s | Gyroscope X |
| `GyrY` | `Float64` | rad/s | Gyroscope Y |
| `GyrZ` | `Float64` | rad/s | Gyroscope Z |
| `AccX` | `Float64` | m/s² | Accelerometer X |
| `AccY` | `Float64` | m/s² | Accelerometer Y |
| `AccZ` | `Float64` | m/s² | Accelerometer Z |

### ATT Message Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | ms | Time since boot |
| `DesRoll` | `Float64` | degrees | Desired roll |
| `Roll` | `Float64` | degrees | Actual roll |
| `DesPitch` | `Float64` | degrees | Desired pitch |
| `Pitch` | `Float64` | degrees | Actual pitch |
| `DesYaw` | `Float64` | degrees | Desired yaw |
| `Yaw` | `Float64` | degrees | Actual yaw |

### Example Analysis

```python
import polars as pl

# Access messages
gps = flight.drone_data['GPS']
att = flight.drone_data['ATT']
bat = flight.drone_data['BAT']

# Convert GPS coordinates
gps = gps.with_columns([
    (pl.col('Lat') / 1e7).alias('lat'),
    (pl.col('Lng') / 1e7).alias('lon'),
    (pl.col('Alt') / 1000).alias('alt_m'),
])

# Find fix quality
fix_counts = gps.group_by('Status').agg(pl.count().alias('count'))
print(fix_counts)

# Battery analysis
bat_stats = bat.select([
    pl.col('Volt').mean().alias('avg_voltage'),
    pl.col('Curr').max().alias('max_current'),
    pl.col('CurrTot').max().alias('total_mah'),
])
print(bat_stats)
```

---

## Litchi

Support for Litchi mission planner exports.

### Format

Litchi CSV contains waypoint mission data:

```csv
latitude,longitude,altitude(m),heading(deg),curvesize(m),rotationdir,
gimbalmode,gimbalpitchangle,actiontype1,actionparam1,...
```

### Loading

```python
from pils.drones import Litchi

# Via flight container
flight.add_drone_data()
litchi_data = flight.drone_data['data']

# Or directly
litchi = Litchi(path="/path/to/mission.csv")
litchi.load_data()
data = litchi.data
```

### Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `latitude` | `Float64` | degrees | Waypoint latitude |
| `longitude` | `Float64` | degrees | Waypoint longitude |
| `altitude` | `Float64` | m | Above ground level |
| `heading` | `Float64` | degrees | Aircraft heading |
| `speed` | `Float64` | m/s | Cruise speed |
| `curve_size` | `Float64` | m | Curve radius at waypoint |
| `rotation_dir` | `Int32` | - | -1=CCW, 0=None, 1=CW |
| `gimbal_mode` | `Int32` | - | 0=Disabled, 1=Focus, 2=Interpolate |
| `gimbal_pitch` | `Float64` | degrees | Gimbal angle |
| `actions` | `Utf8` | - | Action commands (JSON) |

### Example Analysis

```python
import polars as pl

litchi_df = flight.drone_data['data']

# Total mission distance
litchi_df = litchi_df.with_columns([
    pl.col('latitude').diff().alias('lat_diff'),
    pl.col('longitude').diff().alias('lon_diff'),
])

# Count waypoints
print(f"Waypoints: {litchi_df.height}")

# Altitude profile
print(f"Altitude range: {litchi_df['altitude'].min()} - {litchi_df['altitude'].max()} m")
```

## Next Steps

- [Time Synchronization](synchronization.md) - Align drone and sensor data
- [Data Export](data-export.md) - Export processed data
