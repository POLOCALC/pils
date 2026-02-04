# Testing Guide

PILS uses `pytest` for comprehensive testing with 63% code coverage.

## Running Tests

### All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=pils/ --cov-report=html

# Quick run (no verbosity)
pytest tests/ -q
```

### Specific Test Files

```bash
# Run sensor tests only
pytest tests/test_sensors_*.py -v

# Run drone tests
pytest tests/test_drones_*.py -v

# Run specific test file
pytest tests/test_flight_hdf5.py -v
```

### Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_sensors_gps.py::TestGPS -v

# Run specific test function
pytest tests/test_sensors_gps.py::TestGPS::test_load_data_creates_dataframe -v

# Run tests matching pattern
pytest -k "gps" -v
pytest -k "test_load" -v
```

## Test Structure

```
tests/
├── test_correlation_sync.py       # Synchronization tests
├── test_decoders_kernel.py        # KERNEL decoder tests
├── test_drones_blacksquare.py     # BlackSquare drone tests
├── test_drones_dji.py             # DJI drone tests
├── test_drones_litchi.py          # Litchi drone tests
├── test_flight_hdf5.py            # HDF5 storage tests
├── test_path_loader.py            # PathLoader tests
├── test_ppk_analysis.py           # PPK analysis tests
├── test_sensors_adc.py            # ADC sensor tests
├── test_sensors_camera.py         # Camera sensor tests
├── test_sensors_gps.py            # GPS sensor tests
├── test_sensors_imu.py            # IMU sensor tests
├── test_sensors_inclinometer.py   # Inclinometer sensor tests
├── test_stout_loader.py           # StoutLoader tests
└── test_utils_tools.py            # Utility function tests
```

## Coverage Report

Current test coverage: **63%**

### High Coverage Modules (90%+)
- `pils/sensors/adc.py` - 91%
- `pils/sensors/inclinometer.py` - 83%
- `pils/decoders/KERNEL_utils.py` - 83%
- `pils/utils/tools.py` - 100% ✅
- `pils/sensors/IMU.py` - 100% ✅
- `pils/drones/litchi.py` - 100% ✅

### View Coverage HTML Report

```bash
pytest tests/ --cov=pils/ --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## Writing Tests

### Test File Structure

```python
"""
Tests for pils.sensors.gps module.

Following TDD methodology - tests should be written first.
"""

import pytest
import polars as pl
from pathlib import Path
from pils.sensors.gps import GPS


class TestGPS:
    """Test suite for GPS sensor class."""

    @pytest.fixture
    def sample_gps_file(self, tmp_path):
        """Create a sample GPS binary file for testing."""
        gps_file = tmp_path / "sensors" / "gps.bin"
        gps_file.parent.mkdir(parents=True)
        # Create mock binary data
        gps_file.write_bytes(b"\\x00" * 1000)
        return gps_file

    def test_init_with_path(self, sample_gps_file):
        """Test GPS initialization with file path."""
        gps = GPS(sample_gps_file.parent)
        assert gps.path is not None
        assert isinstance(gps.path, Path)

    def test_load_data_creates_dataframe(self, sample_gps_file):
        """Test that load_data returns a Polars DataFrame."""
        gps = GPS(sample_gps_file.parent)
        gps.load_data()
        assert gps.data is not None
        assert isinstance(gps.data, pl.DataFrame)
```

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<what_is_being_tested>`

### Fixtures

Use `pytest.fixture` for reusable test data:

```python
@pytest.fixture
def sample_flight(tmp_path):
    """Create a sample flight structure."""
    flight_dir = tmp_path / "flight_20250101_1200"
    flight_dir.mkdir()
    (flight_dir / "aux").mkdir()
    (flight_dir / "aux" / "sensors").mkdir()
    return flight_dir

@pytest.fixture
def mock_gps_data():
    """Create mock GPS DataFrame."""
    return pl.DataFrame({
        'timestamp': [1000, 2000, 3000],
        'latitude': [40.7128, 40.7129, 40.7130],
        'longitude': [-74.0060, -74.0061, -74.0062],
        'altitude': [10.0, 11.0, 12.0]
    })
```

### Testing with tmp_path

Use `tmp_path` fixture for temporary files:

```python
def test_save_to_hdf5(tmp_path):
    """Test saving flight data to HDF5."""
    output_file = tmp_path / "test_flight.h5"
    
    flight = Flight(flight_info)
    flight.to_hdf5(output_file)
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0
```

### Testing Exceptions

```python
def test_invalid_path_raises_error():
    """Test that invalid path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        GPS("/nonexistent/path")

def test_empty_file_raises_value_error(tmp_path):
    """Test that empty file raises ValueError."""
    empty_file = tmp_path / "empty.bin"
    empty_file.write_bytes(b"")
    
    with pytest.raises(ValueError, match="empty"):
        load_data(empty_file)
```

### Testing Logging

Use `caplog` fixture to test logging:

```python
import logging

def test_warning_logged(caplog):
    """Test that warning is logged."""
    with caplog.at_level(logging.WARNING):
        function_that_logs_warning()
    
    assert "expected warning message" in caplog.text
    assert any("warning" in rec.message.lower() 
               for rec in caplog.records)
```

### Parametrized Tests

Test multiple inputs efficiently:

```python
@pytest.mark.parametrize("input,expected", [
    (32.0, 0.0),      # Freezing point
    (212.0, 100.0),   # Boiling point
    (68.0, 20.0),     # Room temperature
])
def test_fahrenheit_to_celsius(input, expected):
    """Test temperature conversion."""
    result = fahrenheit_to_celsius(input)
    assert abs(result - expected) < 0.01
```

## Continuous Integration

Tests run automatically on:
- Every commit (GitHub Actions)
- Pull requests
- Scheduled daily builds

CI configuration: `.github/workflows/ci.yml`

## Test-Driven Development (TDD)

PILS follows strict TDD methodology:

1. **RED**: Write failing test first
2. **GREEN**: Write minimal code to pass test
3. **REFACTOR**: Clean up code
4. **VERIFY**: Run tests, linters, formatters

Example TDD workflow:

```python
# 1. RED - Write failing test
def test_new_feature():
    """Test new feature that doesn't exist yet."""
    result = new_feature(input_data)
    assert result == expected_output

# Run: pytest tests/test_module.py::test_new_feature -v
# Expected: FAILED (function doesn't exist)

# 2. GREEN - Write minimal implementation
def new_feature(data):
    """Implement new feature."""
    return process(data)

# Run: pytest tests/test_module.py::test_new_feature -v
# Expected: PASSED

# 3. REFACTOR - Clean up code
def new_feature(data: pl.DataFrame) -> pl.DataFrame:
    """Implement new feature with proper types and docs."""
    return data.filter(pl.col('value') > 0)

# 4. VERIFY - Run full test suite
# pytest tests/ -v
# black pils/ tests/
# isort pils/ tests/
# flake8 pils/ tests/
```

## Code Quality Tools

### Black (Code Formatter)

```bash
# Format all code
black pils/ tests/

# Check without modifying
black pils/ tests/ --check

# Format specific file
black pils/sensors/gps.py
```

### isort (Import Sorter)

```bash
# Sort imports
isort pils/ tests/

# Check without modifying
isort pils/ tests/ --check-only
```

### flake8 (Linter)

```bash
# Lint all code
flake8 pils/ tests/

# Lint specific file
flake8 pils/sensors/gps.py

# Ignore specific errors
flake8 pils/ --extend-ignore=E203,W503
```

### mypy (Type Checker)

```bash
# Type check all code
mypy pils/

# Type check specific module
mypy pils/sensors/

# Strict mode
mypy pils/ --strict
```

## Running Full Quality Check

```bash
# Format, sort, lint, test - all in one
black pils/ tests/ && \
isort pils/ tests/ && \
flake8 pils/ tests/ && \
pytest tests/ -v --cov=pils/
```

## Debugging Tests

### Verbose Output

```bash
pytest tests/ -vv --tb=long
```

### Stop on First Failure

```bash
pytest tests/ -x
```

### Run Last Failed Tests

```bash
pytest tests/ --lf
```

### Print Output

```bash
pytest tests/ -s  # Show print() statements
```

### Debug with pdb

```python
def test_something():
    """Test with debugger."""
    import pdb; pdb.set_trace()
    result = function_under_test()
    assert result == expected
```

## Performance Testing

```python
import time

def test_performance():
    """Test that operation completes quickly."""
    start = time.time()
    
    # Operation to test
    process_large_dataset()
    
    duration = time.time() - start
    assert duration < 1.0, f"Too slow: {duration}s"
```

## Best Practices

1. **Test one thing per test** - Each test should verify one behavior
2. **Use descriptive names** - Test name should describe what it tests
3. **Arrange-Act-Assert** - Clear structure: setup, execute, verify
4. **Use fixtures** - Reuse test data and setup code
5. **Mock external dependencies** - Don't depend on network, files, etc.
6. **Test edge cases** - Empty inputs, None values, large datasets
7. **Test error conditions** - Verify exceptions are raised correctly
8. **Keep tests fast** - Fast tests = faster development

## Common Issues

### Import errors

Ensure package is installed in development mode:
```bash
pip install -e .
```

### Fixture not found

Import fixtures or use conftest.py:
```python
# tests/conftest.py
import pytest

@pytest.fixture
def shared_fixture():
    return "shared data"
```

### Async test failures

Use pytest-asyncio plugin:
```bash
pip install pytest-asyncio
```

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result == expected
```

## Next Steps

- Run tests to verify installation: `pytest tests/ -v`
- Check coverage: `pytest tests/ --cov=pils/ --cov-report=html`
- Read [Contributing Guide](contributing.md) for development workflow
- Explore test files in `tests/` directory for examples
