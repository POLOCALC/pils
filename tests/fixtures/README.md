# Test Fixtures

This directory contains test data and fixtures used by the PILS test suite.

## Purpose

Test fixtures provide reusable test data for unit and integration tests. This includes:
- Sample flight data files (CSV, binary formats)
- Mock sensor data (GPS, IMU, ADC, camera, inclinometer)
- Configuration files for testing
- Expected output samples for validation

## Organization

Organize fixtures by type:
- `gps/` - GPS test data
- `imu/` - IMU sensor test data
- `adc/` - ADC sensor test data
- `flights/` - Complete flight data samples
- `configs/` - Test configuration files

## Usage

In pytest tests, fixtures can be loaded using:

```python
from pathlib import Path
import pytest

@pytest.fixture
def sample_gps_csv(tmp_path):
    """Create a sample GPS CSV file."""
    fixture_dir = Path(__file__).parent / "fixtures" / "gps"
    return fixture_dir / "sample_gps.csv"
```

## Guidelines

- Keep fixture files small (< 1MB when possible)
- Use realistic but minimal data samples
- Document expected data format in fixture filenames or comments
- Use `.gitignore` to exclude large binary files if needed
