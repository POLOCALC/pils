# Adding Analysis Modules

Extend PILS with new analysis capabilities.

## Overview

PILS analysis modules are standalone components that process flight data to extract insights. Analysis modules should:

1. Be independent from the Flight class
2. Support versioned execution (track when analysis was run)
3. Store results with `pils.store` (Parquet tables and a manifest per revision)
4. Use Polars DataFrames for data processing
5. Implement smart re-run logic (don't reprocess unchanged data)

---

## Analysis Module Structure

### Core Principles

**Standalone Design:**
```python
# ✓ Good - Independent analysis
analysis = MyAnalysis(flight_path)
result = analysis.run()

# ✗ Bad - Coupled to Flight
flight.run_my_analysis()
```

**Versioned Results:**
```python
# Each run creates a timestamped version
analysis.run(config)  # → rev_20260204_143022
analysis.run(config)  # → rev_20260204_150315 (if config changed)
```

**Persistence with `pils.store`:**
```python
from pils import store

store.save_version(proc / "my_analysis", version_name, {"result_data": df}, metadata)
store.list_versions(proc / "my_analysis")
tables, metadata = store.load_version(proc / "my_analysis", version_name)
```

---

## File Organization & Storage

Every revision is a directory written once (see
[Data Schema and Storage](../user-guide/data-schema.md)):

```
flight_dir/proc/my_analysis/
└── [<parent>/]                      # optional: a telescope, a target, ...
    └── rev_20260204_143022/
        ├── manifest.json            # metadata, code version, tables
        ├── result_data.parquet      # one Parquet file per table
        ├── velocity_plot.png        # plots, reports, logs of the run
        └── statistics.json
```

- Save the tables with `store.save_version(root, version_name, tables,
  metadata, parent=None, kind="my_analysis")`. Plots and other files of the
  run go in the same directory (`root/[parent/]version_name`), before or
  after the tables.
- The manifest is written last and the revision is never changed:
  `save_version` raises `FileExistsError` for a revision that is already
  saved. A new run is a new revision.
- List with `store.list_versions(root, parent)` (oldest first, only complete
  revisions) and load with `store.load_version(root, version_name, parent)`,
  which returns the tables and the metadata.
- Name the columns as in the [schema](../user-guide/data-schema.md):
  `snake_case` with the unit as suffix (`velocity_m_s`, `heading_deg`).
  `store` stores the units in the Parquet files from the names.
- Metadata must be JSON-serialisable (numpy values, paths and datetimes are
  converted).

| Data Type | Where | Format |
|-----------|-------|--------|
| Analysis results (DataFrames) | `rev_*/<table>.parquet` | Parquet |
| Metadata (config, inputs) | `rev_*/manifest.json` | JSON |
| Plots, reports, logs | `rev_*/` | PNG, PDF, JSON, TXT |

!!! warning "Results go through `store.save_version`"
    A CSV written next to the tables is fine for people, but other code reads
    the Parquet tables listed in the manifest.

### Implementation Example

```python
from pils import store


def run_analysis(self, config: dict[str, Any], save: bool = True) -> MyAnalysisVersion:
    """Run the analysis and save it as a new revision."""
    version_name = store.new_version_name()
    result_df = self._execute_analysis(config)
    version = MyAnalysisVersion(
        version_name=version_name,
        result_data=result_df,
        metadata={"config_hash": self._config_hash(config), "config": config},
    )
    if save:
        path = store.save_version(
            self.proc_path,
            version_name,
            {"result_data": result_df},
            version.metadata,
            kind="my_analysis",
        )
        # Files of the run, next to the tables
        self._save_plot(result_df, path / "velocity_plot.png")
    return version


def get_latest_version(self) -> MyAnalysisVersion | None:
    versions = store.list_versions(self.proc_path)
    if not versions:
        return None
    tables, metadata = store.load_version(self.proc_path, versions[-1])
    return MyAnalysisVersion(versions[-1], tables["result_data"], metadata)
```

---

## Analysis Template

### Basic Structure

```python
"""
My analysis module.

Brief description of what this analysis does and why it's useful.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from pils import store

logger = logging.getLogger(__name__)


@dataclass
class MyAnalysisVersion:
    """
    Container for a single analysis revision.
    
    Attributes:
        version_name: Auto-timestamp version (rev_YYYYMMDD_HHMMSS)
        result_data: Analysis results as Polars DataFrame
        metadata: Config hash and execution parameters
        revision_path: Path to revision folder
    """
    version_name: str
    result_data: pl.DataFrame
    metadata: Dict[str, Any]
    revision_path: Path


class MyAnalysis:
    """
    My analysis module with versioned results (pils.store).
    
    This analysis processes flight data to compute [describe output].
    Supports smart re-run logic based on configuration changes.
    
    File Structure:
        flight_dir/proc/my_analysis/
        └── rev_20260204_143022/
            ├── manifest.json             # metadata, code version, tables
            ├── result_data.parquet       # results
            ├── statistics.json           # summary report
            └── config.json               # configuration
    
    Attributes:
        flight_path: Path to flight directory
        proc_path: Path to processing directory (flight_path/proc/my_analysis)
        versions: Dictionary of all analysis versions
        latest_version: Name of most recent version
    
    Example:
        >>> # Create new analysis
        >>> analysis = MyAnalysis('/path/to/flight')
        >>> 
        >>> # Run with configuration
        >>> config = {'param1': 10, 'param2': 'value'}
        >>> version = analysis.run_analysis(config)
        >>> print(version.result_data.head())
        >>> 
        >>> # Smart re-run (only if config changed)
        >>> version2 = analysis.run_analysis(config)  # Skips execution
        >>> 
        >>> # Force re-run
        >>> version3 = analysis.run_analysis(config, force=True)
        >>> 
        >>> # Later: the saved versions are loaded by the constructor
        >>> analysis = MyAnalysis('/path/to/flight')
        >>> latest = analysis.get_latest_version()
    """
    
    def __init__(self, flight_path: str | Path) -> None:
        """
        Initialize analysis for a flight directory.
        
        Args:
            flight_path: Path to flight directory
        """
        self.flight_path = Path(flight_path)
        self.proc_path = self.flight_path / "proc" / "my_analysis"
        self.proc_path.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, MyAnalysisVersion] = {}
        self.latest_version: Optional[str] = None
        
        # Load the saved versions
        for version_name in store.list_versions(self.proc_path):
            tables, metadata = store.load_version(self.proc_path, version_name)
            self.versions[version_name] = MyAnalysisVersion(
                version_name=version_name,
                result_data=tables["result_data"],
                metadata=metadata,
                revision_path=self.proc_path / version_name,
            )
            self.latest_version = version_name
    
    def run_analysis(
        self,
        config: Dict[str, Any],
        force: bool = False
    ) -> MyAnalysisVersion:
        """
        Execute analysis with smart re-run logic.
        
        Only runs if configuration changed or force=True.
        Creates a new timestamped version for each execution.
        
        Args:
            config: Analysis configuration parameters
            force: If True, run even if config unchanged
        
        Returns:
            Analysis version with results
        
        Example:
            >>> config = {'threshold': 0.5, 'window_size': 10}
            >>> version = analysis.run_analysis(config)
            >>> print(f"Version: {version.version_name}")
            >>> print(version.result_data.head())
        """
        # Compute config hash
        config_hash = self._compute_config_hash(config)
        
        # Check if we need to run
        if not force and self._config_exists(config_hash):
            logger.info(f"Config unchanged (hash: {config_hash[:8]}...), skipping execution")
            return self.get_latest_version()
        
        # Generate version name
        version_name = self._generate_version_name()
        revision_path = self.proc_path / version_name
        revision_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Running analysis: {version_name}")
        
        # Execute analysis
        result_data = self._execute_analysis(config, revision_path)
        
        # Save auxiliary data to revision folder (plots, reports, etc.)
        self._save_auxiliary_outputs(result_data, config, revision_path)
        
        # Create version object
        version = MyAnalysisVersion(
            version_name=version_name,
            result_data=result_data,
            metadata={
                'config_hash': config_hash,
                'config': config,
                'execution_time': datetime.now().isoformat(),
            },
            revision_path=revision_path
        )
        
        # Store version
        self.versions[version_name] = version
        self.latest_version = version_name
        
        # Save: tables and manifest next to the auxiliary files
        store.save_version(
            self.proc_path,
            version_name,
            {"result_data": result_data},
            version.metadata,
            kind="my_analysis",
        )
        
        return version
    
    def _execute_analysis(
        self,
        config: Dict[str, Any],
        output_path: Path
    ) -> pl.DataFrame:
        """
        Execute the actual analysis computation.
        
        This is where you implement your analysis logic.
        
        Args:
            config: Analysis parameters
            output_path: Path to save intermediate results
        
        Returns:
            Analysis results as Polars DataFrame
        """
        # Load input data (example: the payload GPS of a saved flight)
        from pils.flight import Flight

        flight = Flight.load(self.flight_path / "proc" / "flight", sync_revision=False)
        df = flight.raw_data.payload_data.gps
        
        # Apply analysis logic
        threshold = config.get('threshold', 0.5)
        window_size = config.get('window_size', 10)
        
        # Example: compute rolling statistics
        # Example: rolling statistics (schema names: unit as suffix)
        result = df.select([
            pl.col('timestamp'),
            pl.col('lat_deg').rolling_mean(window_size).alias('lat_smooth_deg'),
            pl.col('lon_deg').rolling_mean(window_size).alias('lon_smooth_deg'),
            (pl.col('h_acc_m') > threshold).alias('exceeds_threshold')
        ])
        
        return result
    
    def _save_auxiliary_outputs(
        self,
        result_data: pl.DataFrame,
        config: Dict[str, Any],
        output_path: Path
    ) -> None:
        """
        Save auxiliary outputs to revision folder.
        
        This is for plots, reports, logs; the results are saved by
        store.save_version in the same folder.
        
        Args:
            result_data: Analysis results (already in memory)
            config: Configuration used
            output_path: Revision folder path
        """
        import json
        
        # Save configuration as JSON
        with open(output_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Optional: Save summary statistics
        stats = {
            'num_samples': result_data.height,
            'num_exceeded': int(result_data['exceeds_threshold'].sum()),
            'max_lat_deg': float(result_data['lat_smooth_deg'].max()),
        }
        with open(output_path / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Optional: Create a plot (example)
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(result_data['timestamp'], result_data['lat_smooth_deg'])
        # fig.savefig(output_path / "latitude_plot.png")
        # plt.close(fig)
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute SHA256 hash of configuration."""
        import json
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _config_exists(self, config_hash: str) -> bool:
        """Check if this configuration was already run."""
        for version in self.versions.values():
            if version.metadata.get('config_hash') == config_hash:
                return True
        return False
    
    def _generate_version_name(self) -> str:
        """Generate timestamped version name."""
        return datetime.now().strftime("rev_%Y%m%d_%H%M%S")
    
    def get_latest_version(self) -> MyAnalysisVersion:
        """Get most recent analysis version."""
        if not self.latest_version:
            raise ValueError("No analysis versions available")
        return self.versions[self.latest_version]
    
    def get_version(self, version_name: str) -> MyAnalysisVersion:
        """Get specific analysis version by name."""
        if version_name not in self.versions:
            raise KeyError(f"Version not found: {version_name}")
        return self.versions[version_name]
    
    def list_versions(self) -> List[str]:
        """List all available version names in chronological order."""
        return sorted(self.versions.keys())
```

---

## Registration and Integration

### Step 1: Create Module File

Create `pils/analyze/my_analysis.py` with the template above.

### Step 2: Export in __init__.py

Update `pils/analyze/__init__.py`:

```python
from pils.analyze.ppk import PPKAnalysis
from pils.analyze.my_analysis import MyAnalysis

__all__ = ['PPKAnalysis', 'MyAnalysis']
```

### Step 3: Usage Example

```python
from pils.analyze.my_analysis import MyAnalysis

# Initialize
analysis = MyAnalysis('/path/to/flight')

# Run with config
config = {
    'threshold': 0.5,
    'window_size': 10
}
version = analysis.run_analysis(config)

# Access results
print(version.result_data.head())

# Load later: the constructor reads the saved versions
analysis = MyAnalysis('/path/to/flight')
latest = analysis.get_latest_version()
```

---

## Documentation Requirements

When adding a new analysis module or sensor, you **MUST** create documentation in **two places**:

### 1. API Documentation

Create a new file in `docs/api/analysis/` (or `docs/api/sensors/` for sensors) that **automatically extracts docstrings** from your Python code using mkdocstrings.

!!! info "Docstrings are Automatic"
    You **do NOT manually copy docstrings** into the API documentation. The `:::` syntax automatically extracts them from your Python source code. Just reference the module path.

**Example: `docs/api/analysis/my-analysis.md`**

````markdown
# MyAnalysis API

::: pils.analyze.my_analysis.MyAnalysis
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
      
::: pils.analyze.my_analysis.MyAnalysisVersion
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
````

The `:::` directive tells mkdocstrings to:
1. Find the Python class/function
2. Extract its docstring
3. Render it as formatted documentation
4. Optionally show source code (`show_source: true`)

**For Sensors: `docs/api/sensors/barometer.md`**

````markdown
# Barometer Sensor API

::: pils.sensors.barometer.Barometer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 2
````

**Or for entire module:**

````markdown
# Barometer API

::: pils.sensors.barometer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2
````

### 2. User Guide Documentation

Create a comprehensive user guide in `docs/user-guide/` explaining:

- What the analysis/sensor does
- When to use it
- How to use it with examples
- Input/output data formats
- Common use cases

**Example: `docs/user-guide/my-analysis.md`**

```markdown
# My Analysis

Description of what this analysis computes and why it's useful.

## When to Use

- Use case 1
- Use case 2

## Quick Start

\```python
from pils.analyze.my_analysis import MyAnalysis

analysis = MyAnalysis('/path/to/flight')
config = {'threshold': 0.5}
version = analysis.run_analysis(config)
print(version.result_data.head())
\```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.5` | Detection threshold |
| `window_size` | `int` | `10` | Rolling window size |

## Output Schema

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp` | `Float64` | s | UNIX time (UTC) |
| `lat_smooth_deg` | `Float64` | degrees | Smoothed latitude |

## Examples

### Basic Usage

\```python
# Example code
\```

## Next Steps

- [Related Guide](other-page.md)
```

### 3. Update Navigation

Add your new documentation to `mkdocs.yml`:

```yaml
nav:
  - API Reference:
    - Analysis:
      - api/analysis/index.md
      - PPK: api/analysis/ppk.md
      - My Analysis: api/analysis/my-analysis.md  # Add here
  - User Guide:
    - user-guide/index.md
    - My Analysis: user-guide/my-analysis.md  # Add here
```

---

## Complete Example: Velocity Analysis

### File: `pils/analyze/velocity.py`

```python
"""
Velocity Analysis - Compute velocity statistics from GPS data.

Analyzes GPS trajectories to compute instantaneous velocities,
accelerations, and motion statistics.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import polars as pl

from pils import store

logger = logging.getLogger(__name__)


@dataclass
class VelocityVersion:
    """Container for velocity analysis results."""
    version_name: str
    velocity_data: pl.DataFrame
    statistics: Dict[str, float]
    metadata: Dict[str, Any]
    revision_path: Path


class VelocityAnalysis:
    """
    Velocity analysis with smart versioning.
    
    Computes velocities and accelerations from GPS position data.
    
    Example:
        >>> vel = VelocityAnalysis('/path/to/flight')
        >>> version = vel.run_analysis({'smoothing_window': 5})
        >>> print(version.statistics)
        {'max_speed_m_s': 15.3, 'max_accel_m_s2': 2.1, ...}
    """
    
    def __init__(self, flight_path: str | Path) -> None:
        self.flight_path = Path(flight_path)
        self.proc_path = self.flight_path / "proc" / "velocity"
        self.proc_path.mkdir(parents=True, exist_ok=True)
        self.versions: Dict[str, VelocityVersion] = {}
        self.latest_version: Optional[str] = None
    
    def run_analysis(
        self,
        config: Dict[str, Any],
        force: bool = False
    ) -> VelocityVersion:
        """Run velocity analysis."""
        config_hash = self._compute_config_hash(config)
        
        if not force and self._config_exists(config_hash):
            return self.get_latest_version()
        
        version_name = datetime.now().strftime("rev_%Y%m%d_%H%M%S")
        revision_path = self.proc_path / version_name
        revision_path.mkdir(parents=True, exist_ok=True)
        
        # Load the payload GPS of the saved flight
        from pils.flight import Flight

        flight = Flight.load(self.flight_path / "proc" / "flight", sync_revision=False)
        gps = flight.raw_data.payload_data.gps
        
        # Compute velocities
        window = config.get('smoothing_window', 5)
        velocity_data = self._compute_velocities(gps, window)
        
        # Compute statistics
        statistics = {
            'max_speed_m_s': float(velocity_data['speed_m_s'].max()),
            'avg_speed_m_s': float(velocity_data['speed_m_s'].mean()),
            'max_accel_m_s2': float(velocity_data['accel_m_s2'].abs().max()),
        }
        
        version = VelocityVersion(
            version_name=version_name,
            velocity_data=velocity_data,
            statistics=statistics,
            metadata={'config_hash': config_hash, 'config': config},
            revision_path=revision_path
        )
        
        self.versions[version_name] = version
        self.latest_version = version_name
        store.save_version(
            self.proc_path,
            version_name,
            {"velocity_data": velocity_data},
            {**version.metadata, "statistics": statistics},
            kind="velocity",
        )
        
        return version
    
    def _compute_velocities(
        self,
        gps: pl.DataFrame,
        window: int
    ) -> pl.DataFrame:
        """Compute velocities from GPS positions."""
        # Compute differences
        result = gps.select([
            pl.col('timestamp'),
            pl.col('timestamp').diff().alias('dt_s'),
            pl.col('lat_deg').diff().alias('dlat_deg'),
            pl.col('lon_deg').diff().alias('dlon_deg'),
            pl.col('lat_deg'),
        ])
        
        # Convert to meters (approximate)
        R = 6371000  # Earth radius
        result = result.with_columns([
            (pl.col('dlat_deg') * np.pi / 180 * R).alias('dy_m'),
            (
                pl.col('dlon_deg') * np.pi / 180 * R
                * (pl.col('lat_deg') * np.pi / 180).cos()
            ).alias('dx_m'),
        ])
        
        # Speed, smoothed, and acceleration
        result = result.with_columns(
            ((pl.col('dx_m')**2 + pl.col('dy_m')**2).sqrt() / pl.col('dt_s'))
            .rolling_mean(window)
            .alias('speed_m_s')
        )
        result = result.with_columns(
            (pl.col('speed_m_s').diff() / pl.col('dt_s')).alias('accel_m_s2')
        )
        
        return result
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        import json
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    
    def _config_exists(self, config_hash: str) -> bool:
        return any(v.metadata.get('config_hash') == config_hash for v in self.versions.values())
    
    def get_latest_version(self) -> VelocityVersion:
        if not self.latest_version:
            raise ValueError("No versions available")
        return self.versions[self.latest_version]
```

---

## Best Practices

!!! tip "Keep Analysis Standalone"
    Analysis modules should not depend on Flight class. Pass paths, not Flight objects.

!!! tip "Version Everything"
    Always use auto-timestamp versioning (rev_YYYYMMDD_HHMMSS) to track when analysis was run.

!!! tip "Smart Re-run Logic"
    Hash configurations and skip execution if config unchanged (unless force=True).

!!! tip "Use Polars for Performance"
    Polars is 10-100x faster than Pandas for large datasets.

!!! tip "Save with pils.store"
    One write-once revision directory per run (`store.save_version`), with the
    column names of the [schema](../user-guide/data-schema.md).

!!! warning "Document Thoroughly"
    Create both API docs (docstrings) and user guide (examples) for every new module.

---

## Testing

Create tests in `tests/test_my_analysis.py`:

```python
import pytest
import polars as pl
from pathlib import Path
from pils.analyze.my_analysis import MyAnalysis

def test_analysis_execution(tmp_path):
    """Test analysis runs successfully."""
    # Create a saved flight with a mock payload GPS
    from pils.flight import Flight

    flight = Flight({"drone_data_folder_path": str(tmp_path / "drone")})
    flight.raw_data.payload_data.gps = pl.DataFrame({
        'timestamp': [float(i) for i in range(100)],
        'lat_deg': [40.0 + i * 0.0001 for i in range(100)],
        'lon_deg': [-74.0 + i * 0.0001 for i in range(100)],
        'h_acc_m': [1.0] * 100,
    })
    flight.save(tmp_path / "proc" / "flight")
    
    # Run analysis
    analysis = MyAnalysis(tmp_path)
    config = {'threshold': 0.5, 'window_size': 10}
    version = analysis.run_analysis(config)
    
    assert version.result_data.height == 100
    assert 'lat_smooth_deg' in version.result_data.columns

def test_smart_rerun(tmp_path):
    """Test analysis skips unchanged config."""
    # Setup mock data (same as above)
    
    analysis = MyAnalysis(tmp_path)
    config = {'threshold': 0.5}
    
    # First run
    v1 = analysis.run_analysis(config)
    
    # Second run with same config - should skip
    v2 = analysis.run_analysis(config)
    
    assert v1.version_name == v2.version_name

def test_persistence(tmp_path):
    """Test that a saved version loads back."""
    # Setup and run analysis
    
    analysis = MyAnalysis(tmp_path)
    config = {'threshold': 0.5}
    v1 = analysis.run_analysis(config)
    
    # A new instance loads the saved versions
    analysis2 = MyAnalysis(tmp_path)
    v2 = analysis2.get_latest_version()
    
    assert v1.version_name == v2.version_name
    assert v1.result_data.equals(v2.result_data)
```

---

## Next Steps

- [Testing Guide](testing.md) - Write comprehensive tests
- [Code Style](code-style.md) - Follow coding standards
- [Adding Sensors](adding-sensors.md) - Add new sensor types
- [Contributing](contributing.md) - Submit your module
