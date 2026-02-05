# Text Formats

CSV and ASCII text file format specifications.

## CSV Files

### Standard CSV Structure

```csv
timestamp,col1,col2,col3
1700000000000000,1.234,5.678,9.012
1700000000001000,1.235,5.679,9.013
```

### CSV Reading with Polars

```python
import polars as pl

# Basic read
df = pl.read_csv("data.csv")

# With explicit types
df = pl.read_csv("data.csv", dtypes={
    'timestamp': pl.Int64,
    'lat': pl.Float64,
    'lon': pl.Float64,
})

# With null handling
df = pl.read_csv("data.csv", null_values=["", "NA", "null"])

# Lazy loading (large files)
df = pl.scan_csv("data.csv").collect()
```

---

## GPS CSV Formats

### Standard Format

```csv
timestamp,lat,lon,altitude,speed,heading,fix_type,satellite_count
1700000000000000,40.7128,-74.0060,10.5,0.0,0.0,4,12
```

### NMEA-derived Format

```csv
sentence,timestamp,lat,lon,altitude,fix_quality,satellites,hdop
GPGGA,120000.00,4042.7680,N,07400.3600,W,10.5,4,12,1.2
```

### u-blox Format

```csv
iTOW,lon,lat,height,hMSL,hAcc,vAcc,velN,velE,velD
123456000,-740060000,407128000,10500,10200,2500,3500,100,50,-10
```

!!! note "Coordinate Scaling"
    u-blox outputs coordinates as integers (1e-7 degrees). Divide by 10^7 to get decimal degrees.

---

## IMU CSV Formats

### Standard Format

```csv
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
1700000000000000,0.01,-0.02,1.01,0.5,-0.3,0.1
```

### With Magnetometer

```csv
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,temp
1700000000000000,0.01,-0.02,1.01,0.5,-0.3,0.1,25.0,-10.5,45.2,25.5
```

### With Quaternion

```csv
timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,q_w,q_x,q_y,q_z
1700000000000000,0.01,-0.02,1.01,0.5,-0.3,0.1,0.999,0.01,-0.02,0.03
```

---

## DJI SRT Format

Subtitle format embedded in DJI video files.

```
1
00:00:00,000 --> 00:00:01,000
<font size="28">FrameCnt: 1, DiffTime: 33ms
2023.11.01 12:00:00
[iso: 100] [shutter: 1/1000] [fnum: 2.8] [ev: 0] [ct: 5600K]
[latitude: 40.7128] [longitude: -74.0060] [rel_alt: 10.5 abs_alt: 15.2]
[distance: 0.0m] [h_speed: 0.0m/s] [v_speed: 0.0m/s]
</font>

2
00:00:01,000 --> 00:00:02,000
...
```

### Parsing SRT

```python
import re

def parse_srt_frame(text: str) -> dict:
    """Parse a single SRT frame."""
    patterns = {
        'iso': r'\[iso:\s*(\d+)\]',
        'shutter': r'\[shutter:\s*([^\]]+)\]',
        'fnum': r'\[fnum:\s*([\d.]+)\]',
        'latitude': r'\[latitude:\s*([-\d.]+)\]',
        'longitude': r'\[longitude:\s*([-\d.]+)\]',
        'rel_alt': r'\[rel_alt:\s*([-\d.]+)',
        'abs_alt': r'abs_alt:\s*([-\d.]+)\]',
        'h_speed': r'\[h_speed:\s*([-\d.]+)',
        'v_speed': r'\[v_speed:\s*([-\d.]+)',
    }
    
    data = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            data[key] = match.group(1)
    
    return data
```

---

## Litchi CSV Format

```csv
time(millisecond),datetime(utc),latitude,longitude,altitude(m),height_above_takeoff(m),speed(m/s),heading,gimbal_heading,gimbal_pitch,flightmode,message
0,2023-11-01T12:00:00Z,40.7128,-74.0060,15.2,10.5,0.0,90.0,90.0,-45.0,1,
1000,2023-11-01T12:00:01Z,40.7128,-74.0060,15.3,10.6,0.5,92.0,92.0,-45.0,1,
```

---

## RTKLIB POS Format

```
% GPST                  latitude(deg)  longitude(deg)  height(m)   Q  ns  sdn(m)  sde(m)  sdu(m) sdne(m) sdeu(m) sdun(m) age(s) ratio
2023/11/01 12:00:00.000  40.71280000  -74.00600000    10.500      1  12  0.010   0.008   0.020  0.000   0.000   0.000   1.0    5.2
```

### POS Parsing

```python
import polars as pl

def parse_pos_file(filepath: str) -> pl.DataFrame:
    """Parse RTKLIB POS file."""
    # Skip header lines starting with %
    with open(filepath) as f:
        lines = [l for l in f if not l.startswith('%')]
    
    # Parse space-separated values
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 14:
            data.append({
                'datetime': f"{parts[0]} {parts[1]}",
                'lat': float(parts[2]),
                'lon': float(parts[3]),
                'height': float(parts[4]),
                'Q': int(parts[5]),
                'ns': int(parts[6]),
                'sdn': float(parts[7]),
                'sde': float(parts[8]),
                'sdu': float(parts[9]),
                'age': float(parts[12]),
                'ratio': float(parts[13]),
            })
    
    return pl.DataFrame(data)
```

---

## Log Files

### System Log Format

```
2023/11/01 12:00:00.000 [INFO] Recording started
2023/11/01 12:00:01.500 [DEBUG] GPS fix acquired
2023/11/01 12:30:00.000 [INFO] Recording stopped
```

### Parsing Logs

```python
from pils.utils.tools import read_log_time

# Find specific event timestamp
tstart, date = read_log_time(
    keyphrase="Recording started",
    logfile="/data/system.log"
)
```

---

## CSV Writing

```python
import polars as pl

df = pl.DataFrame({
    'timestamp': [1700000000000000, 1700000001000000],
    'lat': [40.7128, 40.7129],
    'lon': [-74.0060, -74.0061],
})

# Write CSV
df.write_csv("output.csv")

# With specific formatting
df.write_csv(
    "output.csv",
    float_precision=6,
    separator=',',
)
```

---

## See Also

- [Binary Formats](binary.md) - Binary file formats
- [HDF5 Format](hdf5.md) - Hierarchical data format
- [GPS Schema](../schemas/gps.md) - GPS data schema
