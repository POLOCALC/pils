"""Aligning the DJI .dat records to GPS time (tick to timestamp)."""

import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class DJIAlignmentMixin:
    """Tick-to-GPS-time alignment for :class:`~pils.drones.DJIDrone.DJIDrone`."""

    @staticmethod
    def _unwrap_tick(df: pl.DataFrame, wrap_threshold: float = 1e8) -> pl.DataFrame:
        """Unwrap tick values that wrap around due to uint32 overflow.

        Only unwraps when tick wraps from high value to near zero (< wrap_threshold).
        Ignores other negative jumps which are likely data corruption.

        Parameters
        ----------
        df : pl.DataFrame
            DataFrame with 'tick' column.
        wrap_threshold : float, optional
            Maximum value after wraparound to be considered valid. Default: 1e8.

        Returns
        -------
        pl.DataFrame
            DataFrame with unwrapped tick values.
        """
        if "tick" not in df.columns or len(df) < 2:
            return df

        tick_values = df["tick"].to_list()
        unwrapped = [tick_values[0]]
        offset = 0

        for i in range(1, len(tick_values)):
            diff = tick_values[i] - tick_values[i - 1]

            # Detect uint32 wraparound: negative jump AND new value is near zero
            if diff < 0 and tick_values[i] < wrap_threshold:
                if 2**32 - wrap_threshold > tick_values[i - 1]:
                    offset += tick_values[i - 1]
                    logger.info(
                        f"Tick resetted at index {i}: {tick_values[i - 1]:,} -> {tick_values[i]:,} (adding offset {tick_values[i - 1]:,}, total offset: {offset:,})"
                    )
                else:
                    offset += 2**32  # Add uint32 max value
                    logger.info(f"{2**32 - 1e4 < tick_values[i - 1]}")
                    logger.info(
                        f"Tick unwrap at index {i}: {tick_values[i - 1]:,} -> {tick_values[i]:,} (adding offset 2^32, total offset: {offset:,})"
                    )

            unwrapped.append(tick_values[i] + offset)

        # Replace tick column with unwrapped values
        return df.with_columns(pl.Series("tick", unwrapped))

    def get_tick_offset(self) -> float:
        """Fit the .dat tick to GPS time and return the offset (s)."""
        # Ensure self.data is a dict for DAT format
        assert isinstance(self.data, dict), "Expected dict for DAT format"

        if "GPS" not in self.data:
            logger.warning("No GPS data available for synchronization")
            return 0.0

        gps_data: pl.DataFrame = self.data["GPS"]

        # Check if we have the required fields
        if "tick" not in gps_data.columns:
            logger.error("GPS data missing 'tick' column")
            return 0.0

        if "timestamp" not in gps_data.columns:
            logger.error("GPS data missing 'timestamp' column")
            return 0.0

        # Calculate the linear regression parameters
        timestamp_arr = gps_data.get_column("timestamp").to_numpy()
        tick_arr = gps_data.get_column("tick").to_numpy()

        diff = np.diff(timestamp_arr)
        (idx,) = np.where(diff > 0.7)

        m, c = np.polyfit(
            tick_arr[idx + 1],
            timestamp_arr[idx + 1],
            1,
        )

        residuals = c + m * tick_arr[idx + 1] - timestamp_arr[idx + 1]
        off = 0

        (idx_fast,) = np.where(residuals[off:] > np.quantile(residuals[off:], 0.95))
        if idx_fast.size == 0:  # equal residuals: no record is late
            (idx_fast,) = np.where(
                residuals[off:] >= np.quantile(residuals[off:], 0.95)
            )

        time_offset = float(
            np.average(
                +(c + m * tick_arr[idx + 1][idx_fast + off])
                - timestamp_arr[idx + 1][idx_fast + off]
            )
        )
        self.data["GPS"] = self.data["GPS"].with_columns(
            pl.Series(
                "timestamp",
                (m * tick_arr + c - time_offset).astype(np.float64),
            )
        )

        self.sync_params = (float(m), time_offset)
        self.tick_to_time = (float(m), float(c - time_offset))
        return time_offset

    def align_datfile(
        self,
        correct_timestamp: bool = True,
        sampling_freq: float = 5.0,
        polars_interpolation: bool = True,
    ) -> pl.DataFrame | None:
        """Align DAT file data using GPS synchronization.

        Parameters
        ----------
        correct_timestamp : bool, optional
            If True, correct timestamps using GPS synchronization.
        sampling_freq : float, optional
            Target sampling frequency in Hz.
        polars_interpolation : bool, optional
            If True, use Polars for interpolation.

        Returns
        -------
        Optional[pl.DataFrame]
            Aligned DataFrame, or None if alignment fails.
        """
        # Ensure self.data is a dict for DAT format
        assert isinstance(self.data, dict), "Expected dict for DAT format"

        if correct_timestamp:
            _ = self.get_tick_offset()

            if polars_interpolation:
                tmp = pl.DataFrame(
                    {
                        "tick": pl.Series([], dtype=pl.Int64),
                        "msg_type": pl.Series([], dtype=pl.Int64),
                    }
                )

                for _i, key in enumerate(self.data):
                    tmp = tmp.join(
                        self.data[key],
                        on=["tick", "msg_type"],
                        how="full",
                        coalesce=True,
                    ).sort("tick")

                numeric_cols = [
                    col
                    for col in tmp.columns
                    if tmp[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                ]
                exclude_cols = {"tick", "msg_type"}

                for col in numeric_cols:
                    if col not in exclude_cols:
                        tmp = tmp.with_columns(
                            pl.col(col).interpolate_by("tick").alias(col)
                        )

                aligned_df = tmp

            else:
                gps_df: pl.DataFrame = self.data["GPS"]
                rtk_df: pl.DataFrame = self.data["RTK"]

                # Determine the start and end ticks based on the overlap of the two datasets
                gps_min = gps_df.get_column("tick").min()
                gps_max = gps_df.get_column("tick").max()
                rtk_min = rtk_df.get_column("tick").min()
                rtk_max = rtk_df.get_column("tick").max()

                if (
                    gps_min is None
                    or gps_max is None
                    or rtk_min is None
                    or rtk_max is None
                ):
                    logger.warning("Could not determine tick range for alignment.")
                    return None

                # Cast to float to ensure numeric comparison
                gps_min_f, gps_max_f = float(gps_min), float(gps_max)  # type: ignore
                rtk_min_f, rtk_max_f = float(rtk_min), float(rtk_max)  # type: ignore

                start_tick = max(gps_min_f, rtk_min_f)
                end_tick = min(gps_max_f, rtk_max_f)

                # Logic limits for the ticks to ensure valid range
                if start_tick >= end_tick:
                    logger.warning("No overlapping data found between GPS and RTK.")
                    return None

                # Create the aligned tick grid based on sampling frequency
                tick_freq = 4_500_000.0
                tick_step = tick_freq / sampling_freq
                target_ticks = np.arange(start_tick, end_tick, tick_step)

                logger.info(
                    f"Target ticks: {len(target_ticks)}, {start_tick:.2f} to {end_tick:.2f}"
                )

                aligned_data: dict[str, np.ndarray] = {"corrected_tick": target_ticks}

                def interpolate_columns(df: pl.DataFrame, exclude_cols: set):
                    # Ensure unique and sorted by corrected_tick for reliable interpolation
                    x = df.get_column("tick").to_numpy()

                    for col in df.columns:
                        if col in exclude_cols:
                            continue

                        # Skip if column is not numeric
                        if df[col].dtype not in [
                            pl.Float32,
                            pl.Float64,
                            pl.Int32,
                            pl.Int64,
                            pl.UInt32,
                            pl.UInt64,
                        ]:
                            continue

                        y = df.get_column(col).to_numpy()
                        try:
                            f = interp1d(
                                x,
                                y,
                                kind="linear",
                                bounds_error=False,
                                fill_value=np.nan,
                            )
                            aligned_data[col] = f(target_ticks)
                        except Exception as e:
                            logger.warning(f"Failed to interpolate column {col}: {e}")

                # Columns to exclude from generic interpolation
                common_exclude = {
                    "tick",
                    "date",
                    "time",
                    "datetime",
                    "timestamp",
                }
                gps_exclude = common_exclude.union({"GPS:date", "GPS:time"})
                rtk_exclude = common_exclude.union({"RTK:date", "RTK:time"})

                interpolate_columns(gps_df, gps_exclude)
                interpolate_columns(rtk_df, rtk_exclude)

                aligned_df = pl.DataFrame(aligned_data)

                aligned_df = aligned_df.with_columns(
                    (pl.col("timestamp") * 1000)
                    .cast(pl.Int64)
                    .cast(pl.Datetime("ms"))
                    .alias("datetime_converted")
                )

                min_val = aligned_data["timestamp"].min()
                max_val = aligned_data["timestamp"].max()

                logger.info(f"Timestamp corrected {min_val}, {max_val}")

                # Ensure timestamp is present and maybe sort columns
                self.aligned_df = aligned_df

        else:
            logger.info("Alignment with no timestamp correction applied")

            base_tick = self.data["GPS"].get_column("tick")[0]

            tmp = pl.DataFrame(
                {
                    "tick": pl.Series([], dtype=pl.Int64),
                    "msg_type": pl.Series([], dtype=pl.Int64),
                }
            )

            for _, key in enumerate(self.data):
                tmp = tmp.join(
                    self.data[key], on=["tick", "msg_type"], how="full", coalesce=True
                ).sort("tick")

            numeric_cols = [
                col
                for col in tmp.columns
                if tmp[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]
            exclude_cols = {"tick", "msg_type"}

            for col in numeric_cols:
                if col not in exclude_cols:
                    tmp = tmp.with_columns(
                        pl.col(col).interpolate_by("tick").alias(col)
                    )

            aligned_df = tmp

            aligned_df = aligned_df.with_columns(
                ((pl.col("tick") - base_tick) / 4_500_000.0).alias("offset")
            )

            timestamp_vals = aligned_df.get_column("timestamp_old").to_numpy()
            tags = np.where(np.diff(timestamp_vals) > 0.5)[0] + 1

            offset_vals = aligned_df.get_column("offset").to_numpy()

            offset_vals = offset_vals[tags].astype(np.float64)
            timestamp_vals = timestamp_vals[tags].astype(np.float64)
            mean_offset = float(np.mean(timestamp_vals - offset_vals))

            aligned_df = aligned_df.with_columns(
                ((pl.col("offset") + mean_offset).cast(pl.Float64)).alias("timestamp")
            )

        return aligned_df
