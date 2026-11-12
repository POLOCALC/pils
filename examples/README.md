# PILS Examples

This folder contains example scripts and notebooks demonstrating PILS functionality.

## Files

- **`loader.ipynb`** - Loading flight data with PathLoader and StoutLoader
- **`gps_comparison_notebook.ipynb`** - GPS data comparison and validation
- **`ppk_stout_analysis.ipynb`** - PPK analysis of flights found through STOUT
- **`quick_azel.ipynb`** - AZEL (azimuth, elevation) of a flight
- **`azel_20251206_1304.csv`** - A sample AZEL output

## Getting Started

```bash
conda activate dm
jupyter notebook loader.ipynb
```

## Quick Example

```python
from pils.flight import Flight
from pils.loader import PathLoader
import polars as pl

# Load flight
loader = PathLoader()
flight = loader.load_single_flight(data_path="/path/to/flight")

# Add sensors
flight.add_sensor_data(['gps', 'imu'])

# Access data
gps_data = flight.raw_data.payload_data.gps
print(gps_data.head())

# Export
flight.to_hdf5("output/flight.h5")
```

## Documentation

For complete documentation, see:
- [Online Documentation](https://polocalc.github.io/pils/)
- [Getting Started Guide](../docs/getting-started/index.md)
- [User Guide](../docs/user-guide/index.md)
- [API Reference](../docs/api/index.md)

## Support

- GitHub Issues: https://github.com/polocalc/pils/issues
- Documentation: https://polocalc.github.io/pils/
