# Installation Guide

## Prerequisites

- Python 3.10 or higher
- Conda package manager (recommended)
- Git

## Installation Methods

### Method 1: Development Installation (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd data_loader
   ```

2. **Create conda environment**:
   ```bash
   conda create -n dm python=3.12
   conda activate dm
   ```

3. **Install dependencies**:
   ```bash
   pip install -e .
   ```

   This installs the package in editable mode, allowing you to modify the source code.

### Method 2: Production Installation

```bash
pip install git+<repository-url>
```

## Dependencies

### Core Dependencies
- **polars >= 0.20.0** - High-performance DataFrame library (primary data structure)
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **pandas** - Data manipulation (legacy support)
- **pyarrow** - Apache Arrow integration

### I/O and Storage
- **h5py** - HDF5 binary storage
- **pyyaml** - YAML configuration files

### Video/Camera
- **opencv-python** - Image and video processing

### Testing (Development)
- **pytest >= 8.0** - Testing framework
- **pytest-cov** - Coverage reporting

### Code Quality (Development)
- **black** - Code formatter
- **isort** - Import sorter
- **flake8** - Linter
- **mypy** - Static type checker

## Verify Installation

```python
import pils
from pils.loader import PathLoader
from pils.flight import Flight

print(f"PILS version: {pils.__version__}")
```

## Development Setup

For contributors, set up the full development environment:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (if available)
pre-commit install

# Run tests to verify setup
pytest tests/ -v
```

## Configuration

### Environment Variables

Set the base path for campaign data (optional):
```bash
export PILS_BASE_PATH="/path/to/campaigns"
```

### Configuration Files

PILS looks for configuration in:
- `~/.pils/config.yml` (user-level)
- `./pils_config.yml` (project-level)

Example configuration:
```yaml
base_path: "/mnt/data/campaigns"
log_level: "INFO"
default_sensors:
  - gps
  - imu
  - camera
```

## Database Setup (Optional)

For StoutLoader (database-based loading):

1. Ensure PostgreSQL/MySQL is running
2. Set connection string:
   ```bash
   export PILS_DB_URI="postgresql://user:pass@localhost/flights"
   ```

## Troubleshooting

### ImportError: No module named 'polars'

Install Polars:
```bash
pip install polars>=0.20.0
```

### OpenCV import fails

Install OpenCV:
```bash
conda install opencv -c conda-forge
```
or
```bash
pip install opencv-python
```

### h5py installation fails

Install HDF5 library first:
```bash
conda install h5py -c conda-forge
```

### Permission denied when installing

Use user installation:
```bash
pip install --user -e .
```

## Platform-Specific Notes

### Linux
No special requirements. Recommended for development.

### macOS
Install Xcode Command Line Tools:
```bash
xcode-select --install
```

### Windows
Consider using WSL (Windows Subsystem for Linux) for best compatibility.

## Next Steps

After installation, proceed to the [Quick Start Guide](quickstart.md).
