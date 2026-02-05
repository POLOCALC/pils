# Installation

Complete guide to installing PILS and its dependencies.

## Requirements

| Requirement | Version | Purpose |
|-------------|---------|---------|
| Python | ≥3.10 | Modern type hints (`str \| Path`) |
| Polars | ≥0.20.0 | High-performance DataFrames |
| NumPy | ≥1.24.0 | Numerical operations |
| h5py | ≥3.8.0 | HDF5 file I/O |
| PyYAML | ≥6.0 | Configuration files |
| stout | ≥ 1.0 | To access database |

## Installation Methods

### Method 1: Conda (Recommended)

```bash
# Create dedicated environment
conda create -n dm python=3.10 -y
conda activate dm

# Clone and install
git clone https://github.com/polocalc/pils.git
cd pils
pip install -e .
```

### Method 2: pip

```bash
pip install pils
```

### Method 3: Development Install

For contributors and developers:

```bash
# Clone repository
git clone https://github.com/polocalc/pils.git
cd pils

# Create conda environment
conda create -n dm python=3.10 -y
conda activate dm

# Install with dev dependencies
pip install -e ".[dev]"
```

## Verifying Installation

```python
# Verify core import
import pils
print(f"PILS version: {pils.__version__}")

# Verify Polars
import polars as pl
print(f"Polars version: {pl.__version__}")

# Verify all modules
from pils.flight import Flight
from pils.loader.path import PathLoader
from pils.sensors import GPS, IMU
print("All modules imported successfully")
```

## Dependencies Explained

### Core Dependencies

| Package | Purpose |
|---------|---------|
| `polars` | Primary DataFrame library, faster than pandas |
| `numpy` | Array operations and numerical computing |
| `h5py` | HDF5 file reading and writing |
| `pyyaml` | YAML configuration file parsing |
| `pathlib` | Modern file path handling (stdlib) |

### Optional Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `opencv-python` | Video/image processing | `pip install opencv-python` |
| `pymavlink` | ArduPilot log parsing | `pip install pymavlink` |
| `matplotlib` | Plotting and visualization | `pip install matplotlib` |
| `rtklib` | PPK processing | System install required |

### Development Dependencies

```bash
pip install -e ".[dev]"
```

Includes:

| Package | Purpose |
|---------|---------|
| `pytest` | Testing framework |
| `pytest-cov` | Coverage reporting |
| `black` | Code formatting |
| `isort` | Import sorting |
| `flake8` | Linting |
| `mypy` | Type checking |


## Environment Variables

Optional environment variables for configuration:

```bash
# Set default campaign path
export PILS_CAMPAIGNS_PATH="/path/to/campaigns"

# Set logging level
export PILS_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR

# Set HDF5 compression
export PILS_HDF5_COMPRESSION="gzip"
export PILS_HDF5_COMPRESSION_LEVEL="4"
```

## Next Steps

Once installed, proceed to the [Quick Start](quickstart.md) guide.
