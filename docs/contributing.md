# Contributing Guide

Thank you for contributing to PILS! This guide will help you set up your development environment and understand our workflow.

## Development Setup

### 1. Clone Repository

```bash
git clone <repository-url>
cd data_loader
```

### 2. Create Development Environment

```bash
conda create -n dm python=3.12
conda activate dm
pip install -e ".[dev]"
```

### 3. Verify Setup

```bash
pytest tests/ -v
```

## Development Workflow

We follow strict **Test-Driven Development (TDD)** methodology.

### TDD Cycle: RED → GREEN → REFACTOR → VERIFY

#### 1. RED: Write Failing Test First

```python
# tests/test_new_feature.py
def test_new_feature():
    """Test new feature (doesn't exist yet)."""
    result = new_feature(input_data)
    assert result == expected_output

# Run test - should FAIL
pytest tests/test_new_feature.py::test_new_feature -v
```

#### 2. GREEN: Write Minimal Code

```python
# pils/module.py
def new_feature(data):
    """Implement new feature."""
    return process(data)

# Run test - should PASS
pytest tests/test_new_feature.py::test_new_feature -v
```

#### 3. REFACTOR: Clean Up Code

```python
# pils/module.py
def new_feature(data: pl.DataFrame) -> pl.DataFrame:
    """Implement new feature with proper types and documentation.
    
    Args:
        data: Input DataFrame with required columns.
        
    Returns:
        Processed DataFrame with new feature.
        
    Raises:
        ValueError: If data is invalid.
    """
    if data.is_empty():
        raise ValueError("Data cannot be empty")
    return data.filter(pl.col('value') > 0)
```

#### 4. VERIFY: Run All Checks

```bash
# Format code
black pils/ tests/
isort pils/ tests/

# Lint
flake8 pils/ tests/

# Type check
mypy pils/

# Run all tests
pytest tests/ -v --cov=pils/
```

## Code Style Guidelines

### Python Style

Follow **PEP 8** with these specifics:

- **Line length**: 100 characters (enforced by Black)
- **Imports**: Sorted by isort
- **Type hints**: Required on all functions
- **Docstrings**: Required on all public functions

### Naming Conventions

```python
# Functions and variables: snake_case
def load_flight_data():
    pass

user_count = 42
is_active = True

# Classes: PascalCase
class GPSDecoder:
    pass

class IMUSensor:
    pass

# Constants: UPPER_SNAKE_CASE
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
SENSOR_CONFIG_PATH = "/etc/pils/sensors.yml"

# Private: _leading_underscore
def _internal_helper():
    pass

_cache = {}
```

### Import Organization

```python
# Standard library first
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Third-party in groups
import numpy as np
import polars as pl
from pydantic import BaseModel

# Local imports last
from .loader import StoutLoader
from .sensors import GPSDecoder

# NO star imports
# Bad: from module import *
# Good: from module import specific_function
```

### Type Hints

```python
from pathlib import Path
from typing import Optional, List, Dict, Any
import polars as pl

# Function with type hints
def load_sensor_data(
    sensor_type: str,
    path: str | Path,
    frequency: Optional[float] = None
) -> pl.DataFrame:
    """Load sensor data from file."""
    pass

# Class with type hints
class Flight:
    def __init__(
        self, 
        flight_info: Dict[str, Any],
        raw_data: Optional[Dict[str, pl.DataFrame]] = None
    ) -> None:
        self.flight_info = flight_info
        self.raw_data = raw_data or {}
```

### Docstrings

Use Google-style docstrings:

```python
def load_sensor_data(sensor_type: str, path: Path) -> pl.DataFrame:
    """Load sensor data from file.
    
    Args:
        sensor_type: Type of sensor (gps, imu, adc, etc.)
        path: Path to sensor data file
        
    Returns:
        Polars DataFrame with sensor data and timestamp column
        
    Raises:
        FileNotFoundError: If sensor file not found
        ValueError: If sensor_type not supported
        
    Example:
        >>> df = load_sensor_data('gps', Path('data/gps.csv'))
        >>> df.shape
        (1000, 5)
    """
    pass
```

## Testing Guidelines

### Test Structure

```python
class TestClassName:
    """Test suite for ClassName."""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample test data."""
        # Setup code
        return data
    
    def test_specific_behavior(self, sample_data):
        """Test specific behavior."""
        # Arrange
        input_data = prepare_input(sample_data)
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected_output
```

### Coverage Requirements

- **New code**: Must have >80% coverage
- **Bug fixes**: Add test that reproduces the bug
- **Refactoring**: Maintain or improve existing coverage

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_sensors_gps.py -v

# Specific test
pytest tests/test_sensors_gps.py::TestGPS::test_load_data -v

# With coverage
pytest tests/ --cov=pils/ --cov-report=html
```

## Code Quality Tools

### Black (Code Formatter)

```bash
# Format code
black pils/ tests/

# Check without modifying
black pils/ tests/ --check
```

Configuration (in `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py310']
```

### isort (Import Sorter)

```bash
# Sort imports
isort pils/ tests/

# Check without modifying
isort pils/ tests/ --check-only
```

Configuration (in `pyproject.toml`):
```toml
[tool.isort]
profile = "black"
line_length = 100
```

### flake8 (Linter)

```bash
# Lint code
flake8 pils/ tests/
```

Configuration (in `.flake8`):
```ini
[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist
```

### mypy (Type Checker)

```bash
# Type check
mypy pils/
```

Configuration (in `pyproject.toml`):
```toml
[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

## Git Workflow

### Branch Naming

- Feature: `feature/description`
- Bug fix: `fix/description`
- Documentation: `docs/description`
- Refactor: `refactor/description`

### Commit Messages

Follow conventional commits:

```
<type>: <short description>

<optional body>

<optional footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `chore`: Maintenance

Examples:
```
feat: add support for IMX-5 inclinometer

- Implement IMX-5 data decoder
- Add auto-detection logic
- Create 15 tests with 90% coverage

Closes #42
```

```
fix: handle missing GPA data in BlackSquare drone

GPA message type not always present in ArduPilot logs.
Make it optional and gracefully handle missing data.

Fixes #58
```

### Pull Request Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/new-sensor
   ```

2. **Implement with TDD**
   - Write tests first (RED)
   - Implement feature (GREEN)
   - Refactor (REFACTOR)
   - Verify (all checks pass)

3. **Commit changes**
   ```bash
   git add .
   git commit -m "feat: add new sensor support"
   ```

4. **Push branch**
   ```bash
   git push origin feature/new-sensor
   ```

5. **Create Pull Request**
   - Fill out PR template
   - Link related issues
   - Request reviewers

6. **Address review feedback**
   - Make requested changes
   - Push updates
   - Mark conversations resolved

7. **Merge when approved**
   - Squash commits if needed
   - Delete branch after merge

## Code Review Checklist

### For Reviewers

- [ ] Code follows TDD methodology
- [ ] Tests cover new functionality (>80%)
- [ ] All tests pass
- [ ] Type hints on all functions
- [ ] Docstrings on public functions
- [ ] No `print()` statements (use logging)
- [ ] Uses `pathlib` (not `os.path`)
- [ ] Polars (not Pandas) for DataFrames
- [ ] Code formatted with Black
- [ ] Imports sorted with isort
- [ ] No linter warnings
- [ ] Documentation updated if needed

### For Contributors

Before requesting review:

```bash
# Run all checks
black pils/ tests/
isort pils/ tests/
flake8 pils/ tests/
mypy pils/
pytest tests/ -v --cov=pils/

# Verify coverage >80% for new code
pytest tests/ --cov=pils/ --cov-report=term-missing
```

## Documentation

### When to Update Docs

- New features → Update relevant `.md` files
- API changes → Update `api_reference.md`
- New examples → Add to `quickstart.md` or `examples/`
- Architecture changes → Update `architecture.md`

### Documentation Structure

```
docs/
├── README.md              # Main documentation index
├── installation.md        # Setup instructions
├── quickstart.md          # Getting started guide
├── architecture.md        # System design
├── loaders.md            # Loader documentation
├── sensors.md            # Sensor documentation
├── drones.md             # Drone platform docs
├── flight.md             # Flight container API
├── synchronization.md    # Synchronization guide
├── ppk_analysis.md       # PPK analysis guide
├── api_reference.md      # Complete API reference
├── testing.md            # Testing guide
└── contributing.md       # This file
```

## Common Tasks

### Adding a New Sensor

1. **Write tests** (`tests/test_sensors_newsensor.py`)
2. **Create sensor class** (`pils/sensors/newsensor.py`)
3. **Register sensor** (update `pils/sensors/sensors.py`)
4. **Update documentation** (`docs/sensors.md`)
5. **Add example** (`examples/newsensor_example.py`)

### Fixing a Bug

1. **Write test that reproduces bug**
2. **Verify test fails** (confirms bug exists)
3. **Fix the bug**
4. **Verify test passes**
5. **Check no regressions** (all tests pass)

### Refactoring Code

1. **Ensure tests exist and pass**
2. **Refactor code**
3. **Verify tests still pass**
4. **Run full quality checks**
5. **Update docs if needed**

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Features**: Open a GitHub Issue with [Feature Request] tag
- **Security**: Email maintainers directly

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Recognition

Contributors are acknowledged in:
- `CONTRIBUTORS.md` file
- Release notes
- Project README

Thank you for contributing to PILS! 🚁
