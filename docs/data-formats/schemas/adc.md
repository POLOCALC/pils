# ADC Schema

Analog-to-Digital Converter data schema reference.

## DataFrame Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Int64` | μs | Unix microseconds |
| `ch0` | `Float64` | V | Channel 0 voltage |
| `ch1` | `Float64` | V | Channel 1 voltage |
| `ch2` | `Float64` | V | Channel 2 voltage |
| `ch3` | `Float64` | V | Channel 3 voltage |
| `ch4` | `Float64` | V | Channel 4 voltage |
| `ch5` | `Float64` | V | Channel 5 voltage |
| `ch6` | `Float64` | V | Channel 6 voltage |
| `ch7` | `Float64` | V | Channel 7 voltage |

!!! note "Channel Count"
    Not all channels may be present. The number of channels depends on the ADC hardware configuration.

---

## Derived Measurements

ADC channels are typically mapped to physical sensors:

| Channel | Typical Mapping | Conversion |
|---------|-----------------|------------|
| ch0 | Pressure sensor | See calibration |
| ch1 | Temperature | V → °C |
| ch2 | Battery voltage | V × divider ratio |
| ch3 | Current sensor | V → A |
| ch4-ch7 | User-defined | Custom |

### Temperature Conversion

```python
# LM35 sensor (10mV/°C)
temp_celsius = voltage * 100

# Thermistor (requires calibration)
# Use Steinhart-Hart equation
```

### Pressure Conversion

```python
# Typical MPXV7002 pressure sensor
# Output: 0.5V at -2kPa, 4.5V at +2kPa
pressure_kpa = (voltage - 2.5) * (4.0 / 4.0)
```

---

## ADC Specifications

### Resolution

| Bits | LSB Voltage (3.3V ref) | LSB Voltage (5V ref) |
|------|------------------------|---------------------|
| 10 | 3.22 mV | 4.88 mV |
| 12 | 0.81 mV | 1.22 mV |
| 16 | 0.05 mV | 0.08 mV |

### Sample Rates

| Rate | Application |
|------|-------------|
| 1 Hz | Slow signals (temperature) |
| 10 Hz | General telemetry |
| 100 Hz | Vibration monitoring |
| 1000+ Hz | High-speed acquisition |

---

## File Format

### CSV Format

```csv
timestamp,ch0,ch1,ch2,ch3
1700000000000000,2.456,1.234,3.300,0.512
1700000000010000,2.458,1.232,3.301,0.514
1700000000020000,2.455,1.235,3.299,0.511
```

### Column Naming

Different systems may use different column names:

| Standard | Alternative |
|----------|-------------|
| `ch0` | `adc0`, `channel_0`, `A0` |
| `ch1` | `adc1`, `channel_1`, `A1` |
| `timestamp` | `time`, `ts`, `t` |

---

## Example

```python
import polars as pl
from pils.sensors.adc import ADC
from pathlib import Path

# Load ADC data
adc = ADC(file=Path("/data/adc.csv"))
df = adc.data

# Check available channels
print(df.columns)

# Convert to physical units (example)
df = df.with_columns([
    # Temperature from ch1 (LM35 sensor)
    (pl.col('ch1') * 100).alias('temperature'),
    
    # Battery voltage from ch2 (1:3 divider)
    (pl.col('ch2') * 3).alias('battery_v'),
])

# Calculate statistics
stats = df.select([
    pl.col('ch0').mean().alias('ch0_mean'),
    pl.col('ch0').std().alias('ch0_std'),
    pl.col('ch0').min().alias('ch0_min'),
    pl.col('ch0').max().alias('ch0_max'),
])
print(stats)

# Detect anomalies
threshold = 4.5  # Volts
anomalies = df.filter(pl.col('ch0') > threshold)
print(f"Anomalies: {anomalies.height}")
```

---

## Calibration

### Linear Calibration

```python
# y = mx + b
def calibrate_linear(voltage: float, m: float, b: float) -> float:
    return m * voltage + b

# Example: Pressure sensor
# 0.5V = -2 kPa, 4.5V = +2 kPa
m = 4.0 / 4.0  # (2 - (-2)) / (4.5 - 0.5)
b = -2.5
pressure = calibrate_linear(voltage, m, b)
```

### Polynomial Calibration

```python
import numpy as np

# Fit polynomial to calibration points
coeffs = np.polyfit(measured_voltages, reference_values, degree=2)
calibrated = np.polyval(coeffs, new_voltage)
```

---

## See Also

- [ADC Sensor API](../api/sensors/adc.md)
- [GPS Schema](gps.md)
- [IMU Schema](imu.md)
