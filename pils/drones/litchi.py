import polars as pl
from ..utils.tools import drop_nan_and_zero_cols


class Litchi:

    def __init__(self, path):
        self.path = path
        self.data = None

    def load_data(
        self,
        cols=[
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
        ],
    ):
        """
        Reads a Litchi flight log file and returns its content as a polars DataFrame.

        Parameters
        ----------
        litchi_path : str
            Path to the Litchi flight log file to be read.

        Returns
        -------
        litchi_data : polars.DataFrame
            DataFrame containing the Litchi flight log data with columns
            ["datetime(utc)", "datetime(local)"] and other relevant data columns.
            The columns contain timestamps in UTC and local time, respectively.
            DataFrame is cleaned of any columns with all NaN or zero values.
        """

        litchi_data = pl.read_csv(self.path, columns=cols)
        litchi_data = litchi_data.with_columns(
            [pl.col("datetime(utc)").str.to_datetime().alias("datetime")]
        )
        litchi_data = litchi_data.drop("datetime(utc)")
        litchi_data = drop_nan_and_zero_cols(litchi_data)
        self.data = litchi_data
