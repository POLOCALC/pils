"""EMLID reference coordinate loader for telescope and base positions."""

from pathlib import Path

import numpy as np
import polars as pl
import pymap3d as pm

from pils.flight import Flight

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class Emlid:
    """Load and process EMLID reference coordinates for telescope tracking.

    This class handles loading telescope and RTK base station positions from
    EMLID survey CSV files. It computes barycenters (geometric mean positions)
    when multiple measurements exist for a single reference point.

    Attributes:
        emlid_path: Path to EMLID CSV file with reference coordinates

    Example:
        >>> from pils.flight import Flight
        >>> flight = Flight(flight_info)
        >>> emlid = Emlid(flight)
        >>> ref_data = emlid.load_data(telescope_name='SATP1')
        >>> telescope_pos = ref_data['telescope']
        >>> base_pos = ref_data['base']['emlid']
    """

    def __init__(self, flight: Flight) -> None:
        """Initialize EMLID coordinate loader from flight path.

        Locates the EMLID reference CSV file in the campaign metadata directory.
        The expected file structure is:
            campaign_dir/metadata/202511_coordinates.csv

        Args:
            flight: Flight object with valid flight_path attribute

        Raises:
            FileNotFoundError: If EMLID CSV file not found at expected path

        Example:
            >>> flight = Flight(flight_info)
            >>> emlid = Emlid(flight)
            >>> print(emlid.emlid_path)
            /path/to/campaign/metadata/202511_coordinates.csv
        """
        flight_path = Path(flight.flight_path)
        campaign_dir = flight_path.parent  # Go up one level to campaign

        self.emlid_path = campaign_dir / "metadata" / "202511_coordinates.csv"

        if not self.emlid_path.exists():
            raise FileNotFoundError(f"EMLID CSV file not found: {self.emlid_path}")

    @staticmethod
    def _get_barycenter(
        coordinates: pl.DataFrame, reference: pl.DataFrame
    ) -> pl.DataFrame:
        """Compute barycenter (geometric mean) of multiple coordinate measurements.

        Converts coordinates to local ENU (East-North-Up) relative to a reference,
        computes the mean ENU position, then converts back to geodetic coordinates.
        This provides a more accurate center position than simple mean lat/lon/alt.

        Args:
            coordinates: DataFrame with columns 'lat', 'lon', 'alt' (multiple rows)
            reference: DataFrame with columns 'lat', 'lon', 'alt' (single row, EMLID base)

        Returns:
            DataFrame with single row containing barycenter {'lat', 'lon', 'alt'}
            in WGS84 degrees and meters ellipsoidal height

        Example:
            >>> coords = pl.DataFrame({
            ...     'lat': [40.0001, 40.0002, 40.0003],
            ...     'lon': [-105.001, -105.002, -105.003],
            ...     'alt': [1500.0, 1500.5, 1501.0]
            ... })
            >>> ref = pl.DataFrame({'lat': [40.0], 'lon': [-105.0], 'alt': [1500.0]})
            >>> barycenter = Emlid._get_barycenter(coords, ref)
            >>> barycenter.shape
            (1, 3)
        """
        # Extract reference position (scalar values from first row)
        ref_lat = reference["lat"][0]
        ref_lon = reference["lon"][0]
        ref_alt = reference["alt"][0]

        # Convert all coordinates to ENU relative to reference
        e, n, u = pm.geodetic2enu(
            coordinates["lat"].to_numpy(),
            coordinates["lon"].to_numpy(),
            coordinates["alt"].to_numpy(),
            ref_lat,
            ref_lon,
            ref_alt,
        )

        # Compute mean position in ENU space
        e_mean = float(np.mean(e))
        n_mean = float(np.mean(n))
        u_mean = float(np.mean(u))

        # Convert mean ENU back to geodetic
        lat, lon, alt = pm.enu2geodetic(
            e_mean,
            n_mean,
            u_mean,
            ref_lat,
            ref_lon,
            ref_alt,
        )

        return pl.DataFrame({"lat": [lat], "lon": [lon], "alt": [alt]})

    def load_data(
        self,
        telescope_name: str,
        base_name: str = "emlid base",
        dji_base_name: str = "dji rtk base (antenna base)",
    ) -> dict[str, pl.DataFrame | dict[str, pl.DataFrame]]:
        """Load telescope and RTK base positions from EMLID reference CSV.

        Reads surveyed positions from EMLID CSV and computes barycenters for
        positions with multiple measurements. The CSV must have columns:
        'Name', 'Longitude', 'Latitude', 'Ellipsoidal height'.

        Args:
            telescope_name: Telescope identifier (e.g., 'SATP1') to filter rows.
                Matches rows where 'Name' starts with this prefix (case-insensitive).
            base_name: EMLID base station identifier (default: 'emlid base').
                Used as reference for barycenter computation.
            dji_base_name: DJI RTK base station identifier
                (default: 'dji rtk base (antenna base)').

        Returns:
            Dictionary with structure:
            {
                'telescope': pl.DataFrame({'lat', 'lon', 'alt'}),  # Single row
                'base': {
                    'emlid': pl.DataFrame({'lat', 'lon', 'alt'}),  # Single row
                    'dji': pl.DataFrame({'lat', 'lon', 'alt'})     # Single row
                }
            }
            All coordinates in WGS84 degrees, altitude in meters (ellipsoidal height).
            If multiple measurements exist, barycenters are computed using ENU.

        Raises:
            FileNotFoundError: If EMLID CSV file not found (checked in __init__)
            ValueError: If telescope or DJI base positions not found in CSV

        Example:
            >>> emlid = Emlid(flight)
            >>> ref_data = emlid.load_data(telescope_name='SATP1')
            >>> telescope_pos = ref_data['telescope'].row(0, named=True)
            >>> print(telescope_pos['lat'], telescope_pos['lon'])
            -22.9597732 -67.7866847
            >>> emlid_base = ref_data['base']['emlid'].row(0, named=True)
            >>> dji_base = ref_data['base']['dji'].row(0, named=True)
        """

        # Load EMLID data with Polars
        df = pl.read_csv(
            self.emlid_path,
            columns=["Name", "Longitude", "Latitude", "Ellipsoidal height"],
        )

        base = {}
        # Get Emlid Base coordinates and transform to lat/lon/alt columns
        emlid_base_df = df.filter(
            pl.col("Name").str.to_lowercase() == base_name.lower()
        )
        if emlid_base_df.height == 0:
            raise ValueError(
                f"No EMLID base position found for '{base_name}' in EMLID CSV"
            )

        emlid_base = emlid_base_df.select(
            [
                pl.col("Latitude").alias("lat"),
                pl.col("Longitude").alias("lon"),
                pl.col("Ellipsoidal height").alias("alt"),
            ]
        )
        base["emlid"] = emlid_base

        # Filter and compute mean for telescope positions
        telescope_df = df.filter(pl.col("Name").str.starts_with(telescope_name.upper()))
        if telescope_df.height == 0:
            raise ValueError(
                f"No telescope positions found for '{telescope_name}' in EMLID CSV"
            )

        telescope = telescope_df.select(
            [
                pl.col("Latitude").alias("lat"),
                pl.col("Longitude").alias("lon"),
                pl.col("Ellipsoidal height").alias("alt"),
            ]
        )

        if telescope.height > 1:
            telescope_barycenter = self._get_barycenter(telescope, emlid_base)
        else:
            telescope_barycenter = telescope.clone()

        # Filter and compute mean for DJI base positions
        dji_df = df.filter(pl.col("Name").str.to_lowercase() == dji_base_name.lower())
        if dji_df.height == 0:
            raise ValueError("No DJI base positions found in EMLID CSV")

        dji = dji_df.select(
            [
                pl.col("Latitude").alias("lat"),
                pl.col("Longitude").alias("lon"),
                pl.col("Ellipsoidal height").alias("alt"),
            ]
        )

        if dji.height > 1:
            dji_barycenter = self._get_barycenter(dji, emlid_base)
        else:
            dji_barycenter = dji.clone()

        base["dji"] = dji_barycenter

        return {"telescope": telescope_barycenter, "base": base}
