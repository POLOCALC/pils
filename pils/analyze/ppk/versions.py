"""A PPK revision (:class:`PPKVersion`) and the tables of files saved before the schema."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


# Position and statistics tables saved before the schema (pils.schema)
LEGACY_POSITION_NAMES = {
    "lat": "lat_deg",
    "lon": "lon_deg",
    "height": "h_ell_m",
    "Q": "quality",
    "ns": "num_sats",
    "sdn": "sd_n_m",
    "sde": "sd_e_m",
    "sdu": "sd_u_m",
    "age": "age_s",
    "east": "e_m",
    "north": "n_m",
    "up": "u_m",
}
LEGACY_STATISTICS_NAMES = {
    "tow": "tow_s",
    "azimuth": "az_deg",
    "elevation": "el_deg",
    "resid_code": "resid_code_m",
    "resid_phase": "resid_phase_m",
    "snr": "snr_dbhz",
}


def legacy_ppk_tables(
    position: pl.DataFrame, statistics: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Position and statistics tables saved before the schema, renamed to it.

    The old position ``time`` was GPS time (the parser only read GPST
    files) and becomes ``datetime`` (UTC) and ``timestamp``; heights were
    ellipsoidal (RTKLIB's default, and the POLOCALC configuration). The old
    statistics had no GPS week: it is taken from the first position epoch,
    and gives their ``timestamp``. Tables already in the schema are returned
    unchanged.
    """
    from pils.analyze.ppkdata.utils import GPS_EPOCH_UNIX, gps_utc_offset, to_utc
    from pils.utils.gnss_time import leap_seconds

    position = position.rename(
        {k: v for k, v in LEGACY_POSITION_NAMES.items() if k in position.columns}
    )
    if "time" in position.columns:
        position = (
            position.with_columns(to_utc(pl.col("time"), "GPST").alias("datetime"))
            .with_columns((pl.col("datetime").dt.epoch("us") / 1e6).alias("timestamp"))
            .select(
                "datetime", "timestamp", pl.exclude("time", "datetime", "timestamp")
            )
        )
    legacy_statistics = "tow" in statistics.columns
    statistics = statistics.rename(
        {k: v for k, v in LEGACY_STATISTICS_NAMES.items() if k in statistics.columns}
    )
    if legacy_statistics:
        if "datetime" in position.columns and position.height and statistics.height:
            first = position["datetime"].min()
            gps_first = (
                first.timestamp()  # type: ignore[union-attr]
                - GPS_EPOCH_UNIX
                + leap_seconds(first)  # only matters at a week boundary
            )
            week0 = int(gps_first // 604800)
            tow0 = gps_first - week0 * 604800
            week = pl.lit(week0) + (pl.col("tow_s") < tow0 - 302400).cast(pl.Int64)
            gps_seconds = week * 604800 + pl.col("tow_s")
            gps_time = pl.from_epoch(
                ((gps_seconds + GPS_EPOCH_UNIX) * 1e6).round().cast(pl.Int64),
                time_unit="us",
            )
            statistics = statistics.with_columns(
                (gps_seconds + GPS_EPOCH_UNIX - gps_utc_offset(gps_time)).alias(
                    "timestamp"
                ),
                week.alias("gps_week"),
            ).select("timestamp", "gps_week", pl.exclude("timestamp", "gps_week"))
    return position, statistics


@dataclass
class PPKVersion:
    """
    Container for a single PPK analysis revision.

    Each revision represents one execution of RTKLIB with specific
    configuration parameters, storing position solutions, statistics,
    and metadata.

    Attributes
    ----------
    version_name : str
        Auto-timestamp version identifier (rev_YYYYMMDD_HHMMSS)
    pos_data : pl.DataFrame
        Position solution DataFrame from .pos file
    stat_data : pl.DataFrame
        Processing statistics DataFrame from .pos.stat file
    metadata : Dict[str, Any]
        Config hash, parsed parameters, and execution metadata
    revision_path : Path
        Path to revision folder containing .pos, .stat, .conf files

    Examples
    --------
    >>> version = PPKVersion(
    ...     version_name='rev_20260204_143022',
    ...     pos_data=pos_df,
    ...     stat_data=stat_df,
    ...     metadata={'config_hash': 'abc123'},
    ...     revision_path=Path('/flight/proc/ppk/rev_20260204_143022')
    ... )
    """

    version_name: str
    pos_data: pl.DataFrame
    stat_data: pl.DataFrame
    metadata: dict[str, Any]
    revision_path: Path
    # Candidate configurations tried by the tuning (run_analysis(optimize=True))
    tuning_data: pl.DataFrame | None = None
