# Code Style

Python coding standards and conventions for PILS.

## Python Version

PILS requires **Python 3.10+** for:

- Modern type hint syntax (`str | None`)
- Pattern matching
- Improved error messages

---

## Formatting

### Tools

| Tool | Purpose | Config |
|------|---------|--------|
| **black** | Code formatting | `pyproject.toml` |
| **isort** | Import sorting | `pyproject.toml` |
| **flake8** | Linting | `.flake8` |
| **mypy** | Type checking | `pyproject.toml` |

### Commands

```bash
# Format code
black pils/ tests/

# Sort imports
isort pils/ tests/

# Lint
flake8 pils/ tests/

# Type check
mypy pils/

# All at once
black pils/ && isort pils/ && flake8 pils/ && mypy pils/
```

---

## Type Hints

### Required on All Functions

```python
# ✅ Good
def load_data(path: str | Path, timeout: int = 30) -> pl.DataFrame:
    pass

# ❌ Bad
def load_data(path, timeout=30):
    pass
```

### Common Types

```python
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import polars as pl

# Basic types
name: str = "flight"
count: int = 42
ratio: float = 0.5
active: bool = True

# Optional (can be None)
data: Optional[pl.DataFrame] = None
data: pl.DataFrame | None = None  # Modern syntax

# Collections
sensors: List[str] = ['gps', 'imu']
config: Dict[str, Any] = {'key': 'value'}

# Path (always use pathlib)
file_path: Path = Path("/data/file.csv")

# Union types
path: str | Path = "/data"

# Return types
def get_data() -> pl.DataFrame: ...
def process() -> None: ...
```

### Generic Types

```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value
    
    def get(self) -> T:
        return self.value
```

---

## Naming Conventions

### Variables and Functions: `snake_case`

```python
# ✅ Good
flight_count = 10
def load_flight_data(): pass
def calculate_average_speed(): pass

# ❌ Bad
flightCount = 10
def loadFlightData(): pass
```

### Classes: `PascalCase`

```python
# ✅ Good
class FlightLoader:
    pass

class GPSDecoder:
    pass

# ❌ Bad
class flight_loader:
    pass
```

### Constants: `UPPER_SNAKE_CASE`

```python
# ✅ Good
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30
GPS_SAMPLING_RATE = 10

# ❌ Bad
maxRetries = 3
default_timeout = 30
```

### Private: `_leading_underscore`

```python
class Flight:
    def __init__(self):
        self._cache = {}  # Private attribute
    
    def _internal_method(self):  # Private method
        pass
```

---

## Imports

### Organization

```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

# 2. Third-party
import numpy as np
import polars as pl
from pydantic import BaseModel

# 3. Local
from pils.flight import Flight
from pils.sensors import GPS
from .utils import helper_function
```

### Rules

```python
# ✅ Good - Explicit imports
from pils.sensors.gps import GPS
from pils.utils.tools import get_path_from_keyword

# ❌ Bad - Star imports
from pils.sensors import *
from pils.utils.tools import *
```

---

## Docstrings

### Google Style

```python
def load_sensor_data(
    sensor_type: str,
    path: Path,
    timeout: int = 30
) -> pl.DataFrame:
    """
    Load sensor data from file.
    
    Args:
        sensor_type: Type of sensor ('gps', 'imu', etc.)
        path: Path to sensor data file
        timeout: Read timeout in seconds
        
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

### Class Docstrings

```python
class GPS:
    """
    GPS sensor data handler.
    
    Reads and processes GPS data from CSV or binary files.
    Supports NMEA and u-blox formats.
    
    Attributes:
        file: Path to GPS data file
        data: Loaded data as Polars DataFrame
        
    Example:
        >>> gps = GPS(file=Path("gps.csv"))
        >>> df = gps.data
        >>> print(df.columns)
        ['timestamp', 'lat', 'lon', 'altitude']
    """
    pass
```

---

## Data Structures

### Polars, Not Pandas

```python
# ✅ Good - Use Polars
import polars as pl

df = pl.read_csv("data.csv")
filtered = df.filter(pl.col('value') > 0)

# ❌ Bad - Avoid Pandas
import pandas as pd
df = pd.read_csv("data.csv")
```

### Pathlib, Not os.path

```python
# ✅ Good - Use pathlib
from pathlib import Path

path = Path("/data/flight")
files = list(path.glob("*.csv"))
content = path.read_text()

# ❌ Bad - Avoid os.path
import os
path = "/data/flight"
files = glob.glob(os.path.join(path, "*.csv"))
```

---

## String Formatting

### F-strings Only

```python
# ✅ Good - F-strings
name = "GPS"
count = 100
print(f"Loaded {count} samples from {name}")

# ❌ Bad - Old formats
print("Loaded %d samples from %s" % (count, name))
print("Loaded {} samples from {}".format(count, name))
```

---

## Error Handling

### Specific Exceptions

```python
# ✅ Good - Catch specific exceptions
try:
    df = pl.read_csv(path)
except FileNotFoundError:
    logger.error(f"File not found: {path}")
    raise
except pl.exceptions.ComputeError as e:
    logger.error(f"Parse error: {e}")
    raise ValueError(f"Invalid CSV format: {path}")

# ❌ Bad - Bare except
try:
    df = pl.read_csv(path)
except:
    pass
```

### Context Managers

```python
# ✅ Good - Use context managers
with open(file_path) as f:
    data = f.read()

with h5py.File(path, 'r') as f:
    df = load_from_hdf5(f)
```

---

## Logging

### Use Logger, Not Print

```python
import logging

logger = logging.getLogger(__name__)

# ✅ Good - Use logger
logger.info(f"Loading flight {flight_id}")
logger.debug(f"Found {len(files)} files")
logger.error(f"Failed to load: {e}", exc_info=True)

# ❌ Bad - Use print
print(f"Loading flight {flight_id}")
```

---

## Classes

### Dataclass for Data Containers

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class FlightInfo:
    flight_id: str
    date: str
    pilot: Optional[str] = None
    location: Optional[str] = None
```

### Properties for Computed Values

```python
class GPS:
    def __init__(self, file: Path):
        self.file = file
        self._data: Optional[pl.DataFrame] = None
    
    @property
    def data(self) -> pl.DataFrame:
        """Lazy-loaded data."""
        if self._data is None:
            self._data = self.load_data()
        return self._data
```

---

## Example: Complete Module

```python
"""
GPS sensor module.

Provides GPS data loading and processing functionality.
"""
import logging
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)


class GPS:
    """
    GPS sensor data handler.
    
    Attributes:
        file: Path to GPS data file
        data: Loaded data as Polars DataFrame
    """
    
    REQUIRED_COLUMNS = ['timestamp', 'lat', 'lon']
    
    def __init__(self, file: Path) -> None:
        """
        Initialize GPS with file path.
        
        Args:
            file: Path to GPS data file
        """
        self.file = Path(file)
        self._data: Optional[pl.DataFrame] = None
    
    @property
    def data(self) -> pl.DataFrame:
        """Get GPS data, loading if necessary."""
        if self._data is None:
            self._data = self.load_data()
        return self._data
    
    def load_data(self) -> pl.DataFrame:
        """
        Load GPS data from file.
        
        Returns:
            GPS data as Polars DataFrame
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns missing
        """
        if not self.file.exists():
            raise FileNotFoundError(f"GPS file not found: {self.file}")
        
        logger.info(f"Loading GPS data from {self.file}")
        
        df = pl.read_csv(self.file)
        
        # Validate columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        logger.debug(f"Loaded {df.shape[0]} GPS samples")
        
        return df
```

---

## See Also

- [Testing](testing.md) - Test guidelines
- [Contributing](contributing.md) - How to contribute
- [Adding Sensors](adding-sensors.md) - Extend PILS
