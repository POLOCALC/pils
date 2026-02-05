# DataFrame Schemas

Reference documentation for all sensor data schemas.

## Overview

PILS uses Polars DataFrames with standardized schemas for each sensor type. This ensures consistent data handling across the system.

## Available Schemas

| Schema | Description |
|--------|-------------|
| [GPS](gps.md) | GPS/GNSS position data |
| [IMU](imu.md) | Inertial measurement unit |
| [ADC](adc.md) | Analog-to-digital converter |
| [Drone](drone.md) | Drone telemetry data |
| [PPK](ppk.md) | Post-processed positions |

## Common Conventions

### Timestamp Format

All timestamps use Unix microseconds:

```python
import polars as pl
from datetime import datetime

# Unix microseconds
timestamp_us: int = 1700000000000000

# Convert to datetime
dt = datetime.fromtimestamp(timestamp_us / 1_000_000)

# In Polars
df = df.with_columns(
    (pl.col('timestamp') / 1_000_000)
    .cast(pl.Datetime('us'))
    .alias('datetime')
)
```

### Coordinate Systems

| System | Description |
|--------|-------------|
| WGS84 | GPS coordinates (degrees, meters) |
| NED | North-East-Down body frame |
| ENU | East-North-Up local frame |

---

## See Also

- [File Formats](../files/binary.md) - Binary and text formats
- [Directory Structure](../directory-structure.md) - Folder layouts
