# Adding Sensors

Extend PILS with new sensor types.

## Overview

PILS uses a sensor registry pattern that makes adding new sensors straightforward:

1. Create sensor class
2. Register in sensor config
3. Use via Flight interface

---

## Sensor Class Template

### Basic Structure

```python
"""
New sensor module.
"""
from pathlib import Path
from typing import Optional

import polars as pl


class NewSensor:
    """
    New sensor data handler.
    
    Attributes:
        file: Path to sensor data file
        data: Loaded sensor data as Polars DataFrame
    """
    
    def __init__(self, file: Path) -> None:
        """
        Initialize sensor with file path.
        
        Args:
            file: Path to sensor data file
        """
        self.file = Path(file)
        self._data: Optional[pl.DataFrame] = None
    
    @property
    def data(self) -> pl.DataFrame:
        """
        Get sensor data, loading if necessary.
        
        Returns:
            Sensor data as Polars DataFrame
        """
        if self._data is None:
            self._data = self.load_data()
        return self._data
    
    def load_data(self) -> pl.DataFrame:
        """
        Load sensor data from file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not self.file.exists():
            raise FileNotFoundError(f"Sensor file not found: {self.file}")
        
        # Load data (customize for your format)
        df = pl.read_csv(self.file)
        
        # Validate and transform
        df = self._validate_schema(df)
        df = self._transform_data(df)
        
        return df
    
    def _validate_schema(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate DataFrame schema."""
        required_columns = ['timestamp', 'value']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    
    def _transform_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply data transformations."""
        # Example: ensure correct types
        df = df.with_columns([
            pl.col('timestamp').cast(pl.Int64),
            pl.col('value').cast(pl.Float64),
        ])
        return df
```

---

## Registration

### Step 1: Create Sensor File

Create `pils/sensors/new_sensor.py`:

```python
from pathlib import Path
from typing import Optional

import polars as pl


class Barometer:
    """Barometer sensor for pressure/altitude data."""
    
    def __init__(self, file: Path) -> None:
        self.file = Path(file)
        self._data: Optional[pl.DataFrame] = None
    
    @property
    def data(self) -> pl.DataFrame:
        if self._data is None:
            self._data = self.load_data()
        return self._data
    
    def load_data(self) -> pl.DataFrame:
        if not self.file.exists():
            raise FileNotFoundError(f"File not found: {self.file}")
        
        df = pl.read_csv(self.file)
        
        # Convert pressure to altitude
        df = df.with_columns([
            ((1 - (pl.col('pressure') / 1013.25) ** 0.190284) * 44330.77)
            .alias('altitude_baro')
        ])
        
        return df
```

### Step 2: Register in Config

Update `pils/sensors/sensors.py`:

```python
from pils.sensors.adc import ADC
from pils.sensors.gps import GPS
from pils.sensors.IMU import IMU
from pils.sensors.inclinometer import Inclinometer
from pils.sensors.barometer import Barometer  # Add import

sensor_config = {
    "gps": {
        "class": GPS,
        "load_method": "load_data",
    },
    "imu": {
        "class": IMU,
        "load_method": "load_all",
    },
    "adc": {
        "class": ADC,
        "load_method": "load_data",
    },
    "inclinometer": {
        "class": Inclinometer,
        "load_method": "load_data",
    },
    # Add new sensor
    "barometer": {
        "class": Barometer,
        "load_method": "load_data",
    },
}
```

### Step 3: Use via Flight

```python
from pils.flight import Flight

flight = Flight(flight_info={'flight_id': 'test'})
flight.add_sensor_data(['barometer'])

baro_data = flight['barometer'].data
print(baro_data.head())
```

---

## Complete Example: Barometer

### File: `pils/sensors/barometer.py`

```python
"""
Barometer sensor module.

Handles pressure and barometric altitude data.
"""
from pathlib import Path
from typing import Optional

import polars as pl


class Barometer:
    """
    Barometer sensor data handler.
    
    Reads pressure data and computes barometric altitude using
    the international standard atmosphere model.
    
    Attributes:
        file: Path to barometer data file
        data: Loaded data as Polars DataFrame
        
    Example:
        >>> baro = Barometer(Path("baro.csv"))
        >>> df = baro.data
        >>> print(df.columns)
        ['timestamp', 'pressure', 'temperature', 'altitude_baro']
    """
    
    # Standard atmosphere constants
    P0 = 1013.25  # hPa (sea level pressure)
    L = 0.0065    # K/m (lapse rate)
    T0 = 288.15   # K (sea level temperature)
    
    def __init__(self, file: Path) -> None:
        """
        Initialize barometer with file path.
        
        Args:
            file: Path to CSV data file
        """
        self.file = Path(file)
        self._data: Optional[pl.DataFrame] = None
    
    @property
    def data(self) -> pl.DataFrame:
        """Get sensor data, loading if necessary."""
        if self._data is None:
            self._data = self.load_data()
        return self._data
    
    def load_data(self) -> pl.DataFrame:
        """
        Load barometer data from CSV file.
        
        Expected columns:
            - timestamp: Unix microseconds
            - pressure: Pressure in hPa
            - temperature: Temperature in Celsius (optional)
        
        Returns:
            DataFrame with computed altitude
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns missing
        """
        if not self.file.exists():
            raise FileNotFoundError(f"Barometer file not found: {self.file}")
        
        df = pl.read_csv(self.file)
        
        # Validate required columns
        if 'pressure' not in df.columns:
            raise ValueError("Missing required column: pressure")
        
        if 'timestamp' not in df.columns:
            raise ValueError("Missing required column: timestamp")
        
        # Compute barometric altitude
        df = df.with_columns([
            pl.col('timestamp').cast(pl.Int64),
            pl.col('pressure').cast(pl.Float64),
            self._pressure_to_altitude(pl.col('pressure')).alias('altitude_baro'),
        ])
        
        return df
    
    @staticmethod
    def _pressure_to_altitude(pressure: pl.Expr) -> pl.Expr:
        """
        Convert pressure to altitude using ISA model.
        
        h = (T0 / L) * (1 - (P / P0)^(R*L / g))
        
        Simplified: h = 44330.77 * (1 - (P / P0)^0.190284)
        """
        return ((1 - (pressure / Barometer.P0) ** 0.190284) * 44330.77)
    
    def calibrate(self, ground_altitude: float) -> pl.DataFrame:
        """
        Calibrate barometric altitude to known ground altitude.
        
        Args:
            ground_altitude: Known altitude at start (meters)
            
        Returns:
            DataFrame with calibrated altitude
        """
        df = self.data
        offset = df['altitude_baro'][0] - ground_altitude
        
        return df.with_columns([
            (pl.col('altitude_baro') - offset).alias('altitude_calibrated')
        ])
```

### Tests: `tests/test_barometer.py`

```python
"""Tests for barometer sensor."""
import pytest
import polars as pl
from pathlib import Path

from pils.sensors.barometer import Barometer


class TestBarometer:
    """Test suite for Barometer sensor."""
    
    @pytest.fixture
    def sample_csv(self, tmp_path: Path) -> Path:
        """Create sample barometer CSV."""
        df = pl.DataFrame({
            'timestamp': [1000000, 2000000, 3000000],
            'pressure': [1013.25, 1010.0, 1005.0],
            'temperature': [20.0, 19.5, 19.0],
        })
        csv_file = tmp_path / "baro.csv"
        df.write_csv(csv_file)
        return csv_file
    
    def test_load_data(self, sample_csv: Path):
        """Test basic data loading."""
        baro = Barometer(file=sample_csv)
        df = baro.data
        
        assert df.shape[0] == 3
        assert 'pressure' in df.columns
        assert 'altitude_baro' in df.columns
    
    def test_altitude_calculation(self, sample_csv: Path):
        """Test altitude calculation from pressure."""
        baro = Barometer(file=sample_csv)
        df = baro.data
        
        # At sea level pressure, altitude should be ~0
        assert abs(df['altitude_baro'][0]) < 1.0
        
        # Lower pressure = higher altitude
        assert df['altitude_baro'][1] > df['altitude_baro'][0]
        assert df['altitude_baro'][2] > df['altitude_baro'][1]
    
    def test_calibration(self, sample_csv: Path):
        """Test altitude calibration."""
        baro = Barometer(file=sample_csv)
        calibrated = baro.calibrate(ground_altitude=100.0)
        
        assert 'altitude_calibrated' in calibrated.columns
        assert abs(calibrated['altitude_calibrated'][0] - 100.0) < 0.1
    
    def test_missing_file_raises(self):
        """Test error for missing file."""
        with pytest.raises(FileNotFoundError):
            Barometer(file=Path("/nonexistent/baro.csv"))
    
    def test_missing_pressure_raises(self, tmp_path: Path):
        """Test error for missing pressure column."""
        csv_file = tmp_path / "bad.csv"
        pl.DataFrame({'timestamp': [1000000]}).write_csv(csv_file)
        
        with pytest.raises(ValueError, match="pressure"):
            baro = Barometer(file=csv_file)
            _ = baro.data
```

---

## Binary Sensors

For binary format sensors, extend the template:

```python
import struct
from pathlib import Path
from typing import Optional, List, Dict

import polars as pl


class BinarySensor:
    """Binary format sensor."""
    
    HEADER = b'\xAA\x55'
    RECORD_SIZE = 16
    
    def __init__(self, file: Path) -> None:
        self.file = Path(file)
        self._data: Optional[pl.DataFrame] = None
    
    @property
    def data(self) -> pl.DataFrame:
        if self._data is None:
            self._data = self.load_data()
        return self._data
    
    def load_data(self) -> pl.DataFrame:
        raw = self.file.read_bytes()
        records = self._parse_binary(raw)
        return pl.DataFrame(records)
    
    def _parse_binary(self, data: bytes) -> List[Dict]:
        records = []
        offset = 0
        
        while offset < len(data) - self.RECORD_SIZE:
            # Find header
            if data[offset:offset+2] != self.HEADER:
                offset += 1
                continue
            
            record = self._parse_record(data[offset:offset+self.RECORD_SIZE])
            records.append(record)
            offset += self.RECORD_SIZE
        
        return records
    
    def _parse_record(self, data: bytes) -> Dict:
        # Example: timestamp (4 bytes) + value (4 bytes float)
        timestamp, value = struct.unpack('<If', data[2:10])
        return {
            'timestamp': timestamp,
            'value': value,
        }
```

---

## Documentation Requirements

When adding a new sensor or analysis module, you **MUST** create documentation in **two places**:

### 1. API Documentation

Create a new file in `docs/api/sensors/` that **automatically extracts docstrings** from your Python code using mkdocstrings.

!!! info "Docstrings are Automatic"
    You **do NOT manually copy docstrings** into the API documentation. The `:::` syntax automatically extracts them from your Python source code. Just reference the module path.

**Example: `docs/api/sensors/barometer.md`**

````markdown
# Barometer Sensor API

::: pils.sensors.barometer.Barometer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
````

**Or for entire module:**

````markdown
# Barometer API

::: pils.sensors.barometer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
````

The `:::` directive tells mkdocstrings to:
1. Find the Python class/function
2. Extract its docstring
3. Render it as formatted documentation
4. Optionally show source code (`show_source: true`)

**For Analysis Modules: `docs/api/analysis/velocity.md`**

````markdown
# Velocity Analysis API

::: pils.analyze.velocity.VelocityAnalysis
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      
::: pils.analyze.velocity.VelocityVersion
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
````

**Or entire module:**

````markdown
# Velocity Analysis API

::: pils.analyze.velocity
    options:
      show_root_heading: true
      heading_level: 2
````

### 2. User Guide Documentation

Create a comprehensive user guide in `docs/user-guide/` explaining:

- What the sensor/analysis does
- When to use it
- How to use it with examples
- Input/output data formats
- Configuration options
- Common use cases

**Example: `docs/user-guide/barometer.md`**

```markdown
# Barometer Sensor

The barometer sensor measures atmospheric pressure and computes barometric altitude.

## Overview

Barometric altitude is computed using the International Standard Atmosphere (ISA) model:

$$
h = \frac{T_0}{L} \left(1 - \left(\frac{P}{P_0}\right)^{\frac{RL}{g}}\right)
$$

Where:
- $h$ = altitude (meters)
- $P$ = measured pressure (hPa)
- $P_0$ = sea level pressure (1013.25 hPa)
- $T_0$ = sea level temperature (288.15 K)
- $L$ = lapse rate (0.0065 K/m)

## When to Use

- **Altitude estimation**: When GPS altitude is noisy or unavailable
- **Vertical velocity**: For climb/descent rate calculations
- **Altitude calibration**: Cross-check with GPS altitude
- **Indoor positioning**: Works without GPS signal

## Quick Start

\```python
from pils.flight import Flight

# Load flight with barometer
flight = Flight.from_path('/path/to/flight')
flight.add_sensor_data(['barometer'])

# Access barometer data
baro_data = flight['barometer'].data
print(baro_data.head())
\```

## Data Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | microseconds | Unix timestamp |
| `pressure` | `Float64` | hPa | Atmospheric pressure |
| `temperature` | `Float64` | °C | Temperature (optional) |
| `altitude_baro` | `Float64` | meters | Computed barometric altitude |

## Calibration

Calibrate to known ground altitude:

\```python
baro = flight['barometer']
ground_altitude = 150.0  # meters

# Calibrate
calibrated = baro.calibrate(ground_altitude)
print(calibrated['altitude_calibrated'])
\```

## Examples

### Altitude Comparison

Compare barometric altitude with GPS altitude:

\```python
import polars as pl

# Get both sensors
baro = flight['barometer'].data
gps = flight['gps'].data

# Merge on timestamp
merged = baro.join(gps, on='timestamp', how='inner')

# Compute difference
merged = merged.with_columns([
    (pl.col('altitude_baro') - pl.col('altitude_gps')).alias('alt_diff')
])

print(f"Mean difference: {merged['alt_diff'].mean():.2f} m")
\```

### Vertical Velocity

Compute climb rate from barometric altitude:

\```python
baro = flight['barometer'].data

# Compute vertical velocity
baro = baro.with_columns([
    pl.col('altitude_baro').diff().alias('dh'),
    pl.col('timestamp').diff().alias('dt'),
])

baro = baro.with_columns([
    (pl.col('dh') / pl.col('dt') * 1e6).alias('vertical_speed')  # m/s
])

print(f"Max climb rate: {baro['vertical_speed'].max():.2f} m/s")
\```

## See Also

- [GPS Sensor](gps.md) - GPS altitude reference
- [Synchronization](synchronization.md) - Align barometer with GPS
- [API Reference](../api/sensors/barometer.md) - Detailed API docs
```

### 3. Update Navigation

Add your new documentation to `mkdocs.yml`:

```yaml
nav:
  - API Reference:
    - Sensors:
      - api/sensors/index.md
      - GPS: api/sensors/gps.md
      - IMU: api/sensors/imu.md
      - Barometer: api/sensors/barometer.md  # Add here
  - User Guide:
    - user-guide/index.md
    - Sensors:
      - Barometer: user-guide/barometer.md  # Add here
```

!!! warning "Documentation is Mandatory"
    When adding a sensor or analysis module, you **MUST** create both API documentation and user guide documentation. This ensures the documentation stays synchronized with the codebase.
    
    **Required Files:**
    - API Doc: `docs/api/sensors/your-sensor.md` or `docs/api/analysis/your-analysis.md`
    - User Guide: `docs/user-guide/your-sensor.md` or `docs/user-guide/your-analysis.md`
    - Navigation: Update `mkdocs.yml` to include new pages

---

## Checklist

When adding a new sensor:

- [ ] Create sensor class in `pils/sensors/`
- [ ] Implement `__init__`, `data` property, `load_data`
- [ ] Add type hints to all methods
- [ ] Write docstrings
- [ ] Register in `sensor_config`
- [ ] Write tests (TDD)
- [ ] **Create API documentation** (`docs/api/sensors/`)
- [ ] **Create user guide** (`docs/user-guide/`)
- [ ] **Update `mkdocs.yml`** navigation
- [ ] Update schema docs

---

## See Also

- [Sensors API](../api/sensors/index.md) - Existing sensors
- [Data Formats](../data-formats/index.md) - Schema reference
- [Testing](testing.md) - Test guidelines
