# PILS - POLOCALC Inertial & Drone Loading System

A comprehensive Python package for loading, decoding, and analyzing flight data from drone missions integrated with the STOUT campaign management system.

## Installation

### Basic Installation
```bash
pip install pils
```

### With STOUT Support
```bash
pip install "pils[stout]"
```

### Development Installation
```bash
git clone https://github.com/POLOCALC/pils.git
cd pils
conda env create -f environment.yml
conda activate dm
pip install -e ".[dev,stout]"
```

## Quick Start

### Load Flight Data

```python
from pils import FlightDataHandler

# Initialize handler
handler = FlightDataHandler(use_stout=True)

# Load a flight
flight = handler.load_flight(flight_id='flight-123')

# Access drone telemetry
drone_data = flight.drone.data
print(drone_data[['latitude', 'longitude', 'altitude']].head())

# Access payload sensors
if flight.payload and flight.payload.gps:
    gps_data = flight.payload.gps.data
    print(gps_data.head())
```

### Load Payload Sensors Only

```python
from pils.datahandler import Payload

# Load all sensors from flight aux folder
payload = Payload(dirpath='/path/to/flight/aux')
payload.load_all()

# Access individual sensors
gps_df = payload.gps.data
imu_df = payload.imu.accelerometer.data
adc_df = payload.adc.data

# Synchronize all sensors to common time base
synchronized = payload.synchronize(target_rate_hz=10.0)
synchronized.write_parquet('synchronized.parquet')
```

### Post-Process GPS with PPK

```python
from pils.analyze.ppk import PPKAnalysis
import polars as pl

# Initialize PPK analysis
ppk = PPKAnalysis(flight_path='/path/to/flight')

# Run RTKLIB post-processing
version = ppk.run_analysis(
    config_path='rtklib.conf',
    rover_obs='rover.obs',
    base_obs='base.obs',
    nav_file='nav.nav'
)

# Access solution data
if version:
    pos_df = version.pos_data
    fixed_solutions = pos_df.filter(pl.col('Q') == 1)
    print(f"Fixed solutions: {len(fixed_solutions)}")
```

## Documentation

Complete documentation is available at: **[https://polocalc.github.io/pils/](https://polocalc.github.io/pils/)**

### Build Documentation Locally

```bash
# Install documentation dependencies
pip install mkdocs mkdocs-material pymdown-extensions mkdocstrings mkdocstrings-python autorefs

# Serve documentation locally
mkdocs serve

# Open http://localhost:8000 in your browser
```

The documentation includes:
- **Getting Started** - Installation and first steps
- **User Guide** - Loading flights, sensors, synchronization, PPK analysis
- **API Reference** - Complete class and method documentation
- **Data Formats** - Schema specifications and file formats
- **Development** - Architecture, testing, and contributing

## Examples

See the `examples/` folder for comprehensive examples:

- `pils_examples.ipynb` - Interactive notebook with all major features
- `ppk_analysis_example.py` - Complete PPK workflow
- `examples.py` - General usage examples

Run examples:

```bash
# Jupyter notebooks
jupyter notebook examples/pils_examples.ipynb

# Python scripts
python examples/ppk_analysis_example.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pils --cov-report=html

# Specific test file
pytest tests/test_sensors_gps.py -v
```

## Development

```bash
# Activate environment
conda activate dm

# Format code
black pils/ tests/

# Lint code
flake8 pils/

# Type check
mypy pils/

# All checks together
black pils/ && isort pils/ && flake8 pils/ && mypy pils/
```

## Data Structure

PILS expects data organized as:

```
base_data_path/
└── campaigns/
    └── Campaign_Name/
        └── 20250115/
            └── flight_20250115_1430/
                ├── drone/               # Drone telemetry
                ├── aux/                 # Payload sensors
                │   ├── sensors/
                │   │   ├── gps.bin
                │   │   ├── accelerometer.bin
                │   │   ├── gyroscope.bin
                │   │   └── ...
                │   └── config.yml
                └── proc/                # Processed data
                    └── ppk/             # PPK analysis results
```

## Key Features

- **STOUT Integration**: Query campaigns and flights from database
- **Multi-Drone Support**: Parse DJI, BlackSquare, and Litchi telemetry
- **Comprehensive Sensors**: GPS, IMU (barometer, accelerometer, gyroscope, magnetometer), ADC, inclinometer, camera
- **Data Synchronization**: Merge all sensors to common time base with interpolation
- **PPK Analysis**: Standalone RTKLIB-based post-processing with version control
- **Polars-Based**: Efficient data processing with polars DataFrames
- **Flexible API**: High-level handler or low-level components for advanced workflows

## Requirements

- Python 3.10+
- polars >= 0.19.0
- STOUT >= 2.0.0 (optional, for database integration)
- RTKLIB (optional, for PPK analysis - requires `rnx2rtkp` binary)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! See [Contributing Guide](docs/development/contributing.md) for details.

## Contact

For issues and questions, please open a GitHub issue or contact the POLOCALC team.
