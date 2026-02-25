from pathlib import Path

import polars as pl

from ..utils.logging_config import get_logger
from ..utils.tools import drop_nan_and_zero_cols

logger = get_logger(__name__)


class Litchi:
    """Loader for Litchi CSV flight logs.

    Handles loading and processing of Litchi waypoint mission flight logs.
    """

    def __init__(self, path: str | Path) -> None:
        """Initialize Litchi loader.

        Parameters
        ----------
        path : Union[str, Path]
            Path to Litchi CSV log file.
        """
        self.path = path
        self.data = None

    def load_data(
        self,
        cols: list[str] | None = None,
    ) -> None:
        """Load Litchi flight log data from CSV file.

        Parameters
        ----------
        cols : Optional[List[str]], optional
            List of columns to load. If None, loads default columns.

        Raises
        ------
        FileNotFoundError
            If CSV file not found.
        """
        if cols is None:
            cols = [
                "latitude",
                "longitude",
                "altitude(m)",
                "speed(mps)",
                "distance(m)",
                "velocityX(mps)",
                "velocityY(mps)",
                "velocityZ(mps)",
                "pitch(deg)",
                "roll(deg)",
                "yaw(deg)",
                "batteryTemperature",
                "pitchRaw",
                "rollRaw",
                "yawRaw",
                "gimbalPitchRaw",
                "gimbalRollRaw",
                "gimbalYawRaw",
                "datetime(utc)",
                "isflying",
            ]

        litchi_data = pl.read_csv(self.path, columns=cols)

        # Parse datetime with timezone format
        litchi_data = litchi_data.with_columns(
            [
                pl.col("datetime(utc)")
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", time_zone="UTC")
                .alias("datetime")
            ]
        )
        litchi_data = litchi_data.drop("datetime(utc)")
        litchi_data = drop_nan_and_zero_cols(litchi_data)
        litchi_data = litchi_data.with_columns(
            (pl.col("datetime").dt.timestamp("ms")).alias("unix_time_ms")
        )
        litchi_data = litchi_data.with_columns(
            (pl.col("unix_time_ms") / 1000.0).alias("timestamp")
        )
        if "gimbalPitchRaw" in litchi_data.columns:
            litchi_data = litchi_data.with_columns(
                (pl.col("gimbalPitchRaw") / 10.0).alias("gimbalPitch")
            )
        self.data = litchi_data
