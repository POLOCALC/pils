# Data Export

Export processed data to various formats.

## Export Formats

| Format | Use Case | Compression | Read Speed |
|--------|----------|-------------|------------|
| HDF5 | Archival, complete flight | Yes | Fast |
| Parquet | Analytics, column-based | Yes | Very Fast |
| CSV | Interchange, human-readable | No | Slow |
| JSON | Web APIs, config | No | Medium |

---

## HDF5 Export

### Export Complete Flight

```python
from pils.flight import Flight

flight = Flight(flight_info)
flight.add_drone_data()
flight.add_sensor_data()

# Export to HDF5
flight.to_hdf5("/path/to/output/flight.h5")
```

### HDF5 Structure

```
flight.h5
├── @metadata
│   ├── flight_name
│   ├── campaign
│   ├── date
│   └── datetime
├── /drone_data
│   └── data
├── /raw_data
│   ├── /gps
│   ├── /imu
│   │   ├── /accelerometer
│   │   ├── /gyroscope
│   │   ├── /magnetometer
│   │   └── /barometer
│   ├── /adc
│   └── /inclinometer
└── /synchronized_data (optional)
```

### Custom Compression

```python
# Default: gzip level 4
flight.to_hdf5("flight.h5")

# Higher compression
flight.to_hdf5(
    "flight.h5",
    compression='gzip',
    compression_level=9
)

# No compression (faster write)
flight.to_hdf5(
    "flight.h5",
    compression=None
)
```

### Reload from HDF5

```python
# Reload complete flight
flight = Flight.from_hdf5("/path/to/flight.h5")

print(f"Flight: {flight.flight_name}")
print(f"Sensors: {list(flight.raw_data.keys())}")
```

---

## Parquet Export

Best for analytics and large datasets.

### Single DataFrame

```python
import polars as pl

gps_df = flight['gps'].data

# Basic export
gps_df.write_parquet("gps.parquet")

# With compression
gps_df.write_parquet(
    "gps.parquet",
    compression='zstd',  # or 'snappy', 'gzip', 'lz4'
    compression_level=3
)
```

### Partitioned Export

For very large datasets:

```python
# Partition by date
gps_df = gps_df.with_columns([
    pl.col('timestamp').cast(pl.Datetime).dt.date().alias('date')
])

gps_df.write_parquet(
    "gps_partitioned/",
    partition_by=['date']
)
```

### Lazy Reading

```python
# Read without loading into memory
gps_lazy = pl.scan_parquet("gps.parquet")

# Filter and aggregate lazily
result = (
    gps_lazy
    .filter(pl.col('fix_quality') == 4)
    .select(['latitude', 'longitude'])
    .collect()  # Execute
)
```

---

## CSV Export

For compatibility and human readability.

### Basic Export

```python
gps_df = flight['gps'].data

# Default (comma-separated)
gps_df.write_csv("gps.csv")

# Tab-separated
gps_df.write_csv("gps.tsv", separator='\t')

# Custom null representation
gps_df.write_csv("gps.csv", null_value='NA')
```

### Datetime Handling

```python
# Format datetime columns
gps_df = gps_df.with_columns([
    pl.col('datetime').dt.strftime('%Y-%m-%d %H:%M:%S.%3f').alias('datetime_str')
])

gps_df.select(['datetime_str', 'latitude', 'longitude', 'altitude']).write_csv("gps.csv")
```

### Streaming Large Files

```python
# Write in batches for memory efficiency
import polars as pl

gps_df = flight['gps'].data
batch_size = 100000

with open("gps_large.csv", "w") as f:
    for i in range(0, gps_df.height, batch_size):
        batch = gps_df.slice(i, batch_size)
        batch.write_csv(f, include_header=(i == 0))
```

---

## JSON Export

For web applications and configuration.

### DataFrame to JSON

```python
# Row-oriented (array of objects)
gps_df.write_json("gps.json", row_oriented=True)

# Column-oriented (object of arrays)
gps_df.write_json("gps.json", row_oriented=False)
```

### Sample Output

=== "Row-oriented"

    ```json
    [
      {"timestamp": 1702034450, "latitude": 40.7128, "longitude": -74.006},
      {"timestamp": 1702034451, "latitude": 40.7129, "longitude": -74.0061}
    ]
    ```

=== "Column-oriented"

    ```json
    {
      "timestamp": [1702034450, 1702034451],
      "latitude": [40.7128, 40.7129],
      "longitude": [-74.006, -74.0061]
    }
    ```

### Metadata Export

```python
import json

# Export flight metadata
metadata = {
    'flight_name': flight.flight_name,
    'date': str(flight.date),
    'campaign': flight.campaign,
    'sensors': list(flight.raw_data.keys()),
    'gps_points': flight['gps'].data.height,
    'duration_seconds': (
        flight['gps'].data['timestamp'].max() - 
        flight['gps'].data['timestamp'].min()
    ) / 1000
}

with open("flight_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

---

## Batch Export

### Export All Flights

```python
from pathlib import Path
from pils.loader.path import PathLoader
from pils.flight import Flight

loader = PathLoader(base_path="/data/campaigns")
output_dir = Path("./exports")
output_dir.mkdir(exist_ok=True)

for flight_info in loader.load_all_flights():
    try:
        flight = Flight(flight_info)
        flight.add_drone_data()
        flight.add_sensor_data(['gps'])
        
        # Export to HDF5
        flight.to_hdf5(output_dir / f"{flight.flight_name}.h5")
        
        # Also export GPS to Parquet
        gps_df = flight['gps'].data
        gps_df.write_parquet(output_dir / f"{flight.flight_name}_gps.parquet")
        
        print(f"✓ {flight.flight_name}")
    except Exception as e:
        print(f"✗ {flight_info['flight_name']}: {e}")
```

### Export Summary Table

```python
import polars as pl

summaries = []
for flight_info in loader.load_all_flights():
    try:
        flight = Flight(flight_info)
        flight.add_sensor_data(['gps'])
        gps_df = flight['gps'].data
        
        summaries.append({
            'flight_name': flight.flight_name,
            'date': str(flight.date),
            'gps_points': gps_df.height,
            'duration_min': (gps_df['timestamp'].max() - gps_df['timestamp'].min()) / 60000,
            'fix_rate': gps_df.filter(pl.col('fix_quality') == 4).height / gps_df.height,
        })
    except:
        pass

summary_df = pl.DataFrame(summaries)
summary_df.write_csv("flight_summary.csv")
print(summary_df)
```

---

## GIS Export

### GeoJSON

```python
import json

gps_df = flight['gps'].data

# Create GeoJSON features
features = []
for row in gps_df.iter_rows(named=True):
    features.append({
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [row['longitude'], row['latitude'], row['altitude']]
        },
        "properties": {
            "timestamp": row['timestamp'],
            "fix_quality": row['fix_quality'],
            "num_satellites": row['num_satellites']
        }
    })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open("flight_track.geojson", "w") as f:
    json.dump(geojson, f)
```

### KML

```python
def to_kml(gps_df, output_path):
    """Export GPS track to KML."""
    coordinates = []
    for row in gps_df.iter_rows(named=True):
        coordinates.append(f"{row['longitude']},{row['latitude']},{row['altitude']}")
    
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Flight Track</name>
    <Placemark>
      <name>GPS Track</name>
      <LineString>
        <coordinates>
          {' '.join(coordinates)}
        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""
    
    with open(output_path, "w") as f:
        f.write(kml)

to_kml(gps_df, "flight_track.kml")
```

---

## Database Export

### SQLite

```python
import sqlite3
import polars as pl

gps_df = flight['gps'].data

# Export to SQLite
conn = sqlite3.connect("flight_data.db")
gps_df.write_database(
    table_name="gps",
    connection=conn,
    if_table_exists="replace"
)
conn.close()
```

### PostgreSQL

```python
# Requires: pip install psycopg2-binary

conn_uri = "postgresql://user:password@localhost:5432/flights"

gps_df.write_database(
    table_name="gps_data",
    connection=conn_uri,
    if_table_exists="append"
)
```

---

## Best Practices

!!! tip "Choose the Right Format"
    - **Archival**: HDF5 (keeps everything together)
    - **Analytics**: Parquet (fast, compressed)
    - **Sharing**: CSV (universal compatibility)
    - **GIS**: GeoJSON/KML (map visualization)

!!! tip "Use Compression"
    Parquet with `zstd` offers the best compression/speed balance.

!!! tip "Include Metadata"
    Always export a metadata file alongside your data.

!!! tip "Verify Exports"
    After export, read back and verify a sample:
    ```python
    # Verify
    gps_df.write_parquet("gps.parquet")
    gps_reloaded = pl.read_parquet("gps.parquet")
    assert gps_df.shape == gps_reloaded.shape
    ```

---

## Next Steps

- [API Reference](../api/index.md) - Complete API documentation
- [Architecture](../development/architecture.md) - System design
