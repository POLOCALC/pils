"""
Synchronizer - Unified data synchronization across multiple data sources.

This module provides the Synchronizer class for synchronizing heterogeneous
sensor data (payload sensors, drone data, etc.) to a common time base.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING
import numpy as np

POLARS_AVAILABLE = False
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    pass

if TYPE_CHECKING:
    from polars import DataFrame as PlDataFrame

logger = logging.getLogger(__name__)


class Synchronizer:
    """
    Synchronizes heterogeneous time-series data to a common time base.

    Supports any data source with a DataFrame containing a timestamp column.
    Can combine payload sensors, drone data, and other time-series sources.

    Attributes:
        data_sources: Dictionary mapping source names to data dictionaries
        synchronized_data: Combined synchronized DataFrame (after calling synchronize())
        time_info: Time range and sample rate information for each source
    """

    def __init__(self):
        """Initialize an empty Synchronizer."""
        self.data_sources: Dict[str, Dict[str, Any]] = {}
        self.synchronized_data: Optional["PlDataFrame"] = None
        self.time_info: Dict[str, Dict[str, Any]] = {}

    def add_source(
        self,
        name: str,
        data: "PlDataFrame",
        timestamp_col: str = "timestamp",
        prefix: Optional[str] = None,
    ) -> None:
        """
        Add a data source for synchronization.

        Args:
            name: Unique identifier for this data source (e.g., 'payload', 'drone_gps')
            data: Polars DataFrame with time-series data
            timestamp_col: Name of the timestamp column in the DataFrame
            prefix: Optional prefix for column names in synchronized output
                   (e.g., 'payload_' for 'payload_gps_lat'). If None, uses name + '_'

        Raises:
            ValueError: If timestamp column doesn't exist or data is empty
        """
        if not POLARS_AVAILABLE:
            raise RuntimeError("Polars is required for synchronization")

        if timestamp_col not in data.columns:
            raise ValueError(
                f"Timestamp column '{timestamp_col}' not found in data for source '{name}'"
            )

        if len(data) == 0:
            raise ValueError(f"Data source '{name}' is empty")

        if name in self.data_sources:
            logger.warning(f"Overwriting existing data source: {name}")

        prefix = prefix or f"{name}_"

        self.data_sources[name] = {
            "data": data,
            "timestamp_col": timestamp_col,
            "prefix": prefix,
        }

        logger.info(
            f"Added data source '{name}' with {len(data)} samples, "
            f"prefix='{prefix}', timestamp_col='{timestamp_col}'"
        )

    def remove_source(self, name: str) -> None:
        """
        Remove a data source.

        Args:
            name: Name of the source to remove
        """
        if name in self.data_sources:
            del self.data_sources[name]
            if name in self.time_info:
                del self.time_info[name]
            logger.info(f"Removed data source: {name}")

    def get_source_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all added data sources.

        Returns:
            Dictionary with source names as keys and info dicts as values.
            Each info dict contains: 'samples', 'duration_s', 'sample_rate_hz',
            't_start', 't_end', 'columns'
        """
        info = {}

        for name, source_info in self.data_sources.items():
            data = source_info["data"]
            timestamp_col = source_info["timestamp_col"]

            if timestamp_col not in data.columns or len(data) < 2:
                logger.warning(
                    f"Source '{name}' has insufficient data for time analysis"
                )
                continue

            timestamps = data[timestamp_col].to_numpy()
            t_start = float(timestamps[0])
            t_end = float(timestamps[-1])
            duration = t_end - t_start

            info[name] = {
                "samples": len(data),
                "duration_s": duration,
                "sample_rate_hz": len(data) / duration if duration > 0 else 0,
                "t_start": t_start,
                "t_end": t_end,
                "columns": data.columns,
            }

        self.time_info = info
        return info

    def get_max_sample_rate(self) -> float:
        """
        Get the maximum sample rate among all sources.

        Returns:
            Maximum sample rate in Hz.
        """
        if not self.time_info:
            self.get_source_info()

        if not self.time_info:
            return 0.0

        return max(s["sample_rate_hz"] for s in self.time_info.values())

    def get_common_time_range(self) -> Tuple[float, float]:
        """
        Get the common time range where all sources have data.

        Returns:
            Tuple of (t_start, t_end) in seconds.
        """
        if not self.time_info:
            self.get_source_info()

        if not self.time_info:
            return (0.0, 0.0)

        t_start = max(s["t_start"] for s in self.time_info.values())
        t_end = min(s["t_end"] for s in self.time_info.values())

        return (t_start, t_end)

    def _interpolate_to_timestamps(
        self,
        data: "PlDataFrame",
        source_col: str,
        target_timestamps: np.ndarray,
        value_cols: List[str],
    ) -> Dict[str, np.ndarray]:
        """
        Interpolate source data to target timestamps.

        Args:
            data: Source DataFrame with time-series data
            source_col: Name of the timestamp column in source data
            target_timestamps: Target timestamps to interpolate to
            value_cols: Columns to interpolate

        Returns:
            Dictionary mapping column names to interpolated values
        """
        source_timestamps = data[source_col].to_numpy()
        result = {}

        for col in value_cols:
            if col in data.columns:
                values = data[col].to_numpy().astype(float)
                # Use numpy interpolation
                interpolated = np.interp(
                    target_timestamps,
                    source_timestamps,
                    values,
                    left=np.nan,
                    right=np.nan,
                )
                result[col] = interpolated

        return result

    def synchronize(
        self,
        target_rate_hz: Optional[float] = None,
        sources: Optional[List[str]] = None,
        source_columns: Optional[Dict[str, List[str]]] = None,
        method: str = "linear",
    ) -> "PlDataFrame":
        """
        Synchronize all data sources to a common time base.

        Creates a unified DataFrame with all source data interpolated to
        a common set of timestamps at the specified rate.

        Args:
            target_rate_hz: Target sample rate in Hz. If None, uses the maximum
                           source rate. Cannot exceed the highest source rate.
            sources: List of source names to include. If None, includes all sources.
            source_columns: Dictionary mapping source names to lists of columns to include.
                           If None, includes all columns except timestamp.
                           Example: {'payload': ['gps_lat', 'gps_lon'], 'drone': ['x', 'y', 'z']}
            method: Interpolation method ('linear' supported, others may be added)

        Returns:
            Polars DataFrame with synchronized data.
            Columns are prefixed with the source prefix (e.g., 'payload_gps_lat', 'drone_x')

        Raises:
            ValueError: If no sources added, target_rate_hz exceeds max rate, or invalid sources
            RuntimeError: If no common time range exists between sources
        """
        if not POLARS_AVAILABLE:
            raise RuntimeError("Polars is required for synchronization")

        if not self.data_sources:
            raise ValueError("No data sources added. Use add_source() first.")

        # Get source info
        info = self.get_source_info()

        # Determine which sources to include
        if sources is None:
            sources = list(info.keys())
        else:
            for source in sources:
                if source not in self.data_sources:
                    raise ValueError(f"Unknown source: {source}")

        if not sources:
            raise ValueError("No valid sources to synchronize")

        # Determine target rate
        max_rate = self.get_max_sample_rate()
        if target_rate_hz is None:
            target_rate_hz = max_rate
        elif target_rate_hz > max_rate:
            raise ValueError(
                f"Target rate {target_rate_hz} Hz exceeds maximum source rate {max_rate:.2f} Hz"
            )

        # Get common time range
        t_start, t_end = self.get_common_time_range()
        if t_end <= t_start:
            raise RuntimeError("No overlapping time range between sources")

        # Generate target timestamps
        n_samples = int((t_end - t_start) * target_rate_hz) + 1
        target_timestamps = np.linspace(t_start, t_end, n_samples)

        # Start building synchronized data
        sync_data = {"timestamp": target_timestamps}

        # Synchronize each source
        for source_name in sources:
            source_info = self.data_sources[source_name]
            data = source_info["data"]
            timestamp_col = source_info["timestamp_col"]
            prefix = source_info["prefix"]

            # Determine which columns to include
            if source_columns and source_name in source_columns:
                cols_to_sync = source_columns[source_name]
            else:
                # Include all columns except timestamp
                cols_to_sync = [c for c in data.columns if c != timestamp_col]

            # Filter to columns that actually exist
            cols_to_sync = [c for c in cols_to_sync if c in data.columns]

            if not cols_to_sync:
                logger.warning(f"No columns to synchronize for source '{source_name}'")
                continue

            # Interpolate
            interpolated = self._interpolate_to_timestamps(
                data, timestamp_col, target_timestamps, cols_to_sync
            )

            # Add to synchronized data with prefix
            for col, values in interpolated.items():
                sync_data[f"{prefix}{col}"] = values

        # Create synchronized DataFrame
        self.synchronized_data = pl.DataFrame(sync_data)  # type: ignore[possibly-undefined]

        logger.info(
            f"Synchronized {len(sources)} source(s) to {target_rate_hz:.2f} Hz "
            f"({n_samples} samples, {t_end - t_start:.2f}s duration)"
        )

        return self.synchronized_data

    def save_synchronized(self, output_path: str, format: str = "parquet") -> None:
        """
        Save synchronized data to file.

        Args:
            output_path: Path to output file
            format: Output format ('parquet', 'csv', 'json')

        Raises:
            RuntimeError: If no synchronized data available
        """
        if self.synchronized_data is None:
            raise RuntimeError("No synchronized data. Call synchronize() first.")

        if format == "parquet":
            self.synchronized_data.write_parquet(output_path)
        elif format == "csv":
            self.synchronized_data.write_csv(output_path)
        elif format == "json":
            self.synchronized_data.write_json(output_path)
        else:
            raise ValueError(
                f"Unknown format: {format}. Use 'parquet', 'csv', or 'json'."
            )

        logger.info(f"Saved synchronized data to {output_path}")

    def summary(self) -> str:
        """
        Get a summary of all data sources and their properties.

        Returns:
            Formatted string with source information.
        """
        info = self.get_source_info()
        if not info:
            return "No data sources added."

        lines = ["Synchronizer Data Sources Summary", "=" * 60]

        for name, data in info.items():
            lines.append(f"\n{name.upper()}")
            lines.append(f"  Samples: {data['samples']}")
            lines.append(f"  Duration: {data['duration_s']:.2f} s")
            lines.append(f"  Sample Rate: {data['sample_rate_hz']:.2f} Hz")
            lines.append(f"  Time Range: {data['t_start']:.3f} - {data['t_end']:.3f} s")
            lines.append(f"  Columns: {', '.join(data['columns'])}")

        t_start, t_end = self.get_common_time_range()
        max_rate = self.get_max_sample_rate()

        lines.append("\n" + "=" * 60)
        lines.append(
            f"Common Time Range: {t_start:.3f} - {t_end:.3f} s ({t_end - t_start:.2f} s)"
        )
        lines.append(f"Maximum Sample Rate: {max_rate:.2f} Hz")

        return "\n".join(lines)
