"""Surveyed positions of the telescopes and the RTK bases (EMLID survey).

Notes
-----
Each campaign can have its survey in ``<campaign>/metadata/*_coordinates.csv``
(EMLID export: ``Name``, ``Longitude``, ``Latitude``, ``Ellipsoidal height``).
:func:`reference_positions` finds the positions AZEL needs:

| Position | Source, in order |
|----------|------------------|
| Telescope | the campaign's survey; the newest survey of another campaign that has it (the telescopes do not move); with ``telescope_source="so3g"``, the ``so3g`` site (``so3g.proj.SITES``, used by sotodlib) |
| DJI RTK base | the campaign's survey only (the base is set up again each campaign); without it the drone RTK track is not corrected |

The ``so3g`` sites are given to the arcsecond with one elevation for the
whole site (5188 m): 2-3 m horizontally and about 3 m vertically from the
November 2025 survey, about 0.5 deg of azimuth and elevation for a drone
300 m away. They are used only when asked for.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

    Attributes
    ----------
    emlid_path : Path
        Path to EMLID CSV file with reference coordinates.

    Examples
    --------
    >>> from pils.flight import Flight
    >>> flight = Flight(flight_info)
    >>> emlid = Emlid(flight)
    >>> ref_data = emlid.load_data(telescope_name='SATP1')
    >>> telescope_pos = ref_data['telescope']
    >>> base_pos = ref_data['base']['emlid']
    """

    def __init__(
        self, flight: Flight | None = None, path: str | Path | None = None
    ) -> None:
        """Initialize EMLID coordinate loader from flight path.

        Locates the EMLID reference CSV file in the campaign metadata directory
        by searching for files matching the pattern "*_coordinates.csv".

        The expected file structure is:
            campaign_dir/metadata/<YYYYMM>_coordinates.csv

        Parameters
        ----------
        flight : Flight, optional
            Flight object with valid flight_path attribute.
        path : str or Path, optional
            A coordinates file to read instead of the flight's campaign file.

        Raises
        ------
        FileNotFoundError
            If no EMLID CSV file found or multiple matches found.

        Examples
        --------
        >>> flight = Flight(flight_info)
        >>> emlid = Emlid(flight)
        >>> print(emlid.emlid_path)
        /path/to/campaign/metadata/202511_coordinates.csv
        """
        if path is not None:
            self.emlid_path = Path(path)
            return
        if flight is None:
            raise ValueError("Give a flight or the path of a coordinates file")
        flight_path = Path(flight.flight_path)
        campaign_dir = flight_path.parents[1]  # Go up two levels to campaign

        metadata_dir = campaign_dir / "metadata"

        # Find coordinate files matching pattern
        if not metadata_dir.exists():
            raise FileNotFoundError(
                f"EMLID CSV file not found: metadata directory does not exist at {metadata_dir}"
            )

        coord_files = list(metadata_dir.glob("*_coordinates.csv"))

        if len(coord_files) == 0:
            raise FileNotFoundError(
                f"EMLID CSV file not found: no files matching '*_coordinates.csv' in {metadata_dir}"
            )
        elif len(coord_files) > 1:
            file_list = ", ".join(f.name for f in coord_files)
            raise FileNotFoundError(
                f"EMLID CSV file not found: multiple coordinate files found in {metadata_dir}: {file_list}. "
                "Expected exactly one file matching '*_coordinates.csv'."
            )

        self.emlid_path = coord_files[0]

    @staticmethod
    def _get_barycenter(
        coordinates: pl.DataFrame, reference: pl.DataFrame
    ) -> pl.DataFrame:
        """Compute barycenter (geometric mean) of multiple coordinate measurements.

        Converts coordinates to local ENU (East-North-Up) relative to a reference,
        computes the mean ENU position, then converts back to geodetic coordinates.
        This provides a more accurate center position than simple mean lat/lon/alt.

        Parameters
        ----------
        coordinates : pl.DataFrame
            DataFrame with columns 'lat', 'lon', 'alt' (multiple rows).
        reference : pl.DataFrame
            DataFrame with columns 'lat', 'lon', 'alt' (single row, EMLID base).

        Returns
        -------
        pl.DataFrame
            DataFrame with single row containing barycenter {'lat', 'lon', 'alt'}
            in WGS84 degrees and meters ellipsoidal height.

        Examples
        --------
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

        Parameters
        ----------
        telescope_name : str
            Telescope identifier (e.g., 'SATP1') to filter rows.
            Matches rows where 'Name' starts with this prefix (case-insensitive).
        base_name : str, optional
            EMLID base station identifier.
            Used as reference for barycenter computation.
            Default is 'emlid base'.
        dji_base_name : str, optional
            DJI RTK base station identifier.
            Default is 'dji rtk base (antenna base)'.

        Returns
        -------
        dict[str, pl.DataFrame | dict[str, pl.DataFrame]]
            Dictionary with structure::

                {
                    'telescope': pl.DataFrame({'lat', 'lon', 'alt'}),  # Single row
                    'base': {
                        'emlid': pl.DataFrame({'lat', 'lon', 'alt'}),  # Single row
                        'dji': pl.DataFrame({'lat', 'lon', 'alt'})     # Single row
                    }
                }

            All coordinates in WGS84 degrees, altitude in meters (ellipsoidal height).
            If multiple measurements exist, barycenters are computed using ENU.

        Raises
        ------
        FileNotFoundError
            If EMLID CSV file not found (checked in __init__).
        ValueError
            If telescope or DJI base positions not found in CSV.

        Examples
        --------
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

        if emlid_base.height > 1:
            # For emlid base barycenter, use first measurement as initial reference
            initial_ref = emlid_base.head(1)
            emlid_base_barycenter = self._get_barycenter(emlid_base, initial_ref)
        else:
            emlid_base_barycenter = emlid_base.clone()

        base["emlid"] = emlid_base_barycenter

        # Filter and compute mean for telescope positions
        telescope_df = df.filter(
            pl.col("Name").str.to_lowercase().str.starts_with(telescope_name.lower())
        )
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


TELESCOPE_SOURCES = ("survey", "so3g")
"""Where :func:`reference_positions` takes the telescope position from."""

SO3G_SITES = {
    "satp1": "so_sat1",
    "satp2": "so_sat2",
    "satp3": "so_sat3",
    "lat": "so_lat",
}
"""Telescope names and their ``so3g.proj.SITES`` keys."""


@dataclass
class ReferencePositions:
    """Telescope and DJI RTK base positions, with where they come from.

    Positions are dicts with ``lat``, ``lon`` (deg) and ``alt`` (m,
    ellipsoidal height).

    Attributes
    ----------
    telescope : dict
        Telescope position.
    telescope_source : str
        ``"campaign survey"``, ``"survey <YYYYMM>"`` or ``"so3g <site>"``.
    dji_base : dict or None
        Surveyed DJI RTK base of the campaign; None if not surveyed.
    telescope_file, base_file : str or None
        The coordinates files used.
    notes : list of str
        What the user should know (positions not from the campaign).
    """

    telescope: dict[str, float]
    telescope_source: str
    dji_base: dict[str, float] | None = None
    telescope_file: str | None = None
    base_file: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_metadata(self) -> dict[str, Any]:
        """Return the sources for the AZEL metadata."""
        return {
            "telescope_position_source": self.telescope_source,
            "telescope_position_file": self.telescope_file,
            "base_position_source": "campaign survey" if self.dji_base else None,
            "base_position_file": self.base_file,
            "position_notes": list(self.notes),
        }


def coordinate_files(campaign_dir: str | Path) -> list[Path]:
    """List the survey files: the campaign's, then the others, newest first.

    Parameters
    ----------
    campaign_dir : str or Path
        ``.../campaigns/<YYYYMM>``; the other campaigns are its siblings.

    Returns
    -------
    list of Path
        ``metadata/*_coordinates.csv`` files.
    """
    campaign_dir = Path(campaign_dir)
    own = sorted((campaign_dir / "metadata").glob("*_coordinates.csv"))
    others = [
        f
        for folder in sorted(campaign_dir.parent.iterdir(), reverse=True)
        if folder.is_dir() and folder != campaign_dir
        for f in sorted((folder / "metadata").glob("*_coordinates.csv"))
    ]
    return own + others


def _barycenter(df: pl.DataFrame, prefix: str, exact: bool) -> dict[str, float] | None:
    name = pl.col("Name").str.to_lowercase()
    rows = df.filter(
        name == prefix.lower() if exact else name.str.starts_with(prefix.lower())
    )
    if rows.height == 0:
        return None
    rows = rows.select(
        pl.col("Latitude").alias("lat"),
        pl.col("Longitude").alias("lon"),
        pl.col("Ellipsoidal height").alias("alt"),
    )
    if rows.height > 1:
        rows = Emlid._get_barycenter(rows, rows.head(1))
    return {k: float(v) for k, v in rows.row(0, named=True).items()}


def _read_survey(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path, columns=["Name", "Longitude", "Latitude", "Ellipsoidal height"]
    )


def so3g_position(telescope_name: str) -> tuple[dict[str, float], str]:
    """Return the ``so3g`` site of a telescope and its key.

    Raises
    ------
    ValueError
        If the telescope has no ``so3g`` site.
    ImportError
        If ``so3g`` (installed with sotodlib) is not installed.
    """
    key = SO3G_SITES.get(telescope_name.lower())
    if key is None:
        raise ValueError(
            f"No so3g site for {telescope_name} (known: {sorted(SO3G_SITES)})"
        )
    from so3g.proj import SITES  # type: ignore[import-not-found]

    site = SITES[key]
    return {
        "lat": float(site.lat),
        "lon": float(site.lon),
        "alt": float(site.elev),
    }, key


def reference_positions(
    campaign_dir: str | Path,
    telescope_name: str,
    telescope_source: str = "survey",
    dji_base_name: str = "dji rtk base (antenna base)",
) -> ReferencePositions:
    """Find the telescope and DJI RTK base positions for a campaign.

    Parameters
    ----------
    campaign_dir : str or Path
        ``.../campaigns/<YYYYMM>``.
    telescope_name : str
        E.g. ``"SATP1"``: survey rows whose name starts with it.
    telescope_source : {"survey", "so3g"}, default="survey"
        ``"survey"``: the campaign's survey, else the newest survey of another
        campaign. ``"so3g"``: the so3g site (2-3 m, see the module notes).
    dji_base_name : str, default="dji rtk base (antenna base)"
        Survey name of the DJI RTK base.

    Returns
    -------
    ReferencePositions
        The positions and their sources.

    Raises
    ------
    ValueError
        If no survey has the telescope (with ``"survey"``), or the source is
        unknown.
    """
    if telescope_source not in TELESCOPE_SOURCES:
        raise ValueError(f"telescope_source must be one of {TELESCOPE_SOURCES}")
    campaign_dir = Path(campaign_dir)
    files = coordinate_files(campaign_dir)
    own = [f for f in files if f.parent.parent == campaign_dir]

    dji_base, base_file = None, None
    for f in own:
        dji_base = _barycenter(_read_survey(f), dji_base_name, exact=True)
        if dji_base is not None:
            base_file = str(f)
            break

    notes: list[str] = []
    if telescope_source == "so3g":
        so3g_geod, key = so3g_position(telescope_name)
        notes.append(
            f"{telescope_name} position from so3g ({key}): about 2-3 m horizontally "
            "and 3 m vertically from the survey"
        )
        found: ReferencePositions | None = ReferencePositions(
            so3g_geod, f"so3g {key}", notes=notes
        )
    else:
        found = None
        for f in files:
            position = _barycenter(_read_survey(f), telescope_name, exact=False)
            if position is None:
                continue
            if f.parent.parent == campaign_dir:
                source = "campaign survey"
            else:
                source = f"survey {f.parent.parent.name}"
                notes.append(
                    f"{telescope_name} position from the {f.parent.parent.name} survey "
                    f"({f.name}); this campaign has no survey of it"
                )
            found = ReferencePositions(
                position, source, telescope_file=str(f), notes=notes
            )
            break
        if found is None:
            raise ValueError(
                f"No telescope positions for '{telescope_name}' in any campaign survey "
                f"({len(files)} files); telescope_source='so3g' uses the so3g site "
                "(2-3 m off)"
            )
    assert found is not None
    found.dji_base, found.base_file = dji_base, base_file
    if dji_base is None:
        found.notes.append(
            "No surveyed DJI RTK base in this campaign: the drone RTK track is not "
            "corrected for the base position"
        )
    return found
