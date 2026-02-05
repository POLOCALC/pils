# PILS Examples

This folder contains example scripts and notebooks demonstrating PILS functionality.

## Files

### Jupyter Notebooks

- **`pils_examples.ipynb`** - Comprehensive examples notebook covering all major features:
  - Loading flight data
  - Working with sensors (GPS, IMU, ADC)
  - Processing drone telemetry
  - Time synchronization
  - PPK analysis
  - Data export
  - HDF5 operations
  - Advanced trajectory analysis

- **`loader.ipynb`** - Examples of loading flight data with PathLoader and StoutLoader

- **`gps_comparison_notebook.ipynb`** - GPS data comparison and validation

- **`test.ipynb`** - Testing and debugging notebook

- **`test_hdf5_methods.ipynb`** - HDF5 save/load method testing

### Python Scripts

- **`ppk_analysis_example.py`** - Complete PPK analysis workflow
- **`examples.py`** - General PILS usage examples
- **`compare_dat_csv.py`** - DAT vs CSV file comparison
- **`compare_gps_csv.py`** - GPS data format comparison

## Getting Started

### Running Notebooks

```bash
# Activate conda environment
conda activate dm

# Install Jupyter
conda install jupyter

# Start Jupyter
jupyter notebook

# Open pils_examples.ipynb
```

### Running Scripts

```bash
# Activate environment
conda activate dm

# Run PPK analysis example
python ppk_analysis_example.py

# Run general examples
python examples.py
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
gps_data = flight['gps'].data
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
