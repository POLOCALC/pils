import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from stout.services import CampaignService, set_silent_mode
from stout.config import Config

# Enable silent logging mode for data_loader package usage
set_silent_mode(True)


class PathHandler:
    """
    PathHandler for retrieving flight data paths from stout database or direct paths.

    This class can either query the stout database using CampaignService to retrieve
    file paths for a specific flight name, or work directly with file system paths.
    """

    def __init__(self, flight_name_or_path: str, db_manager=None):
        """
        Initialize PathHandler with a flight name or direct path.

        Automatically detects whether to load from database or filesystem:
        - If the string is a valid path that exists on the filesystem, uses direct path loading
        - Otherwise, treats it as a flight name and loads from the database

        Parameters
        ----------
        flight_name_or_path : str
            Either a flight name (e.g., 'flight_20240715_1430') for database loading,
            or a direct path to the flight data folder for filesystem loading
        db_manager : optional
            Database manager instance (used for database loading)
        """
        self.flight_name_or_path = flight_name_or_path

        # Auto-detect: if path exists on filesystem, use direct loading
        path_obj = Path(flight_name_or_path)
        self.from_db = not path_obj.exists()

        self.db_manager = db_manager

        # Flight metadata (only populated if from_db=True)
        self.flight_data = None
        self.campaign_data = None
        self.payload_data = None

        # File paths
        self.drone_data_folder = None
        self.aux_data_folder = None
        self.processed_data_folder = None
        self.so_obs_paths = None
        self.class_obs_paths = None

        # Loaded flag
        self._loaded = False

    def load_flight_data(self):
        """
        Load flight data from the database or direct paths.

        This method automatically determines whether to query the database or load from
        filesystem based on whether flight_name_or_path exists as a path.

        Returns
        -------
        bool
            True if paths were loaded successfully, False otherwise
        """
        try:
            if self.from_db:
                # Load from database - treat input as flight name
                print(
                    f"[PathHandler] Loading from database: '{self.flight_name_or_path}'"
                )
                campaign_service = CampaignService(self.db_manager)
                flight = campaign_service.get_flight(
                    flight_name=self.flight_name_or_path
                )

                if not flight:
                    print(
                        f"[PathHandler] Flight with name '{self.flight_name_or_path}' not found in database"
                    )
                    return False

                self.flight_data = flight

                # Get campaign data
                campaign_id = flight.get("campaign_id")
                if campaign_id:
                    self.campaign_data = campaign_service.get_campaign_by_id(
                        campaign_id
                    )

                # Get payload data
                payload_id = flight.get("payload_id")
                if payload_id:
                    self.payload_data = campaign_service.get_payload_by_id(payload_id)

                # Extract file paths
                self.drone_data_folder = flight.get("drone_data_folder_path")
                self.aux_data_folder = flight.get("aux_data_folder_path")
                self.processed_data_folder = flight.get("processed_data_folder_path")
                self.so_obs_paths = flight.get("so_obs_id")
                self.class_obs_paths = flight.get("class_obs_id")

                print(
                    f"[PathHandler] Successfully loaded flight '{self.flight_name_or_path}' from database"
                )
                print(
                    f"  Campaign: {self.campaign_data.get('name') if self.campaign_data else 'N/A'}"
                )
                print(
                    f"  Payload: {self.payload_data.get('name') if self.payload_data else 'N/A'}"
                )
                if self.flight_data:
                    print(f"  Takeoff: {flight.get('takeoff_datetime')}")
                    print(f"  Landing: {flight.get('landing_datetime')}")
            else:
                # Load from direct path - treat input as filesystem path
                print(
                    f"[PathHandler] Loading from filesystem path: '{self.flight_name_or_path}'"
                )
                base_path = Path(self.flight_name_or_path).resolve()

                if not base_path.exists():
                    print(f"[PathHandler] Path does not exist: {base_path}")
                    return False

                # Assume standard folder structure: base_path/drone and base_path/aux
                self.drone_data_folder = (
                    str(base_path / "drone") if (base_path / "drone").exists() else None
                )
                self.aux_data_folder = (
                    str(base_path / "aux") if (base_path / "aux").exists() else None
                )
                self.processed_data_folder = (
                    str(base_path / "proc") if (base_path / "proc").exists() else None
                )

                print(f"[PathHandler] Successfully loaded paths from: {base_path}")
                print(f"  Drone folder: {self.drone_data_folder}")
                print(f"  Aux folder: {self.aux_data_folder}")
                print(f"  Processed folder: {self.processed_data_folder}")

            self._loaded = True
            return True

        except Exception as e:
            print(f"[PathHandler] Error loading flight data: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_drone_files(self, pattern: Optional[str] = None):
        """
        Get list of drone data files from drone_data_folder.

        Parameters
        ----------
        pattern : str, optional
            Glob pattern to filter files (e.g., "*.csv", "FLY*.csv")

        Returns
        -------
        list of str
            List of file paths matching the pattern
        """
        if not self._loaded:
            if not self.load_flight_data():
                return []

        if not self.drone_data_folder or not os.path.exists(self.drone_data_folder):
            print(
                f"[PathHandler] Drone data folder not found: {self.drone_data_folder}"
            )
            return []

        import glob

        if pattern:
            search_path = os.path.join(self.drone_data_folder, pattern)
            files = glob.glob(search_path)
        else:
            files = [
                os.path.join(self.drone_data_folder, f)
                for f in os.listdir(self.drone_data_folder)
                if os.path.isfile(os.path.join(self.drone_data_folder, f))
            ]

        return sorted(files)

    def get_aux_files(self, pattern: Optional[str] = None):
        """
        Get list of auxiliary data files from aux_data_folder (recursive).

        Parameters
        ----------
        pattern : str, optional
            Glob pattern to filter files (e.g., "*.log", "**/*.csv")
            If no pattern provided, returns all files recursively

        Returns
        -------
        list of str
            List of file paths matching the pattern
        """
        if not self._loaded:
            if not self.load_flight_data():
                return []

        if not self.aux_data_folder or not os.path.exists(self.aux_data_folder):
            print(f"[PathHandler] Aux data folder not found: {self.aux_data_folder}")
            return []

        import glob

        if pattern:
            search_path = os.path.join(self.aux_data_folder, pattern)
            files = glob.glob(search_path, recursive=True)
        else:
            # Get all files recursively
            search_path = os.path.join(self.aux_data_folder, "**", "*")
            all_files = glob.glob(search_path, recursive=True)
            # Filter to only files (not directories)
            files = [f for f in all_files if os.path.isfile(f)]

        return sorted(files)

    def get_so_observations(self, telescope: Optional[str] = None):
        """
        Get SO telescope observation paths.

        Parameters
        ----------
        telescope : str, optional
            Specific telescope to get observations for ('SATp1', 'SATp2', 'SATp3')
            If None, returns all observations

        Returns
        -------
        str or dict
            If telescope specified: path string or "Not available"
            If telescope=None: dict with all telescope paths
        """
        if not self._loaded:
            if not self.load_flight_data():
                return "Not available" if telescope else {}

        if not self.so_obs_paths:
            return "Not available" if telescope else {}

        # If so_obs_paths is a dict (new format)
        if isinstance(self.so_obs_paths, dict):
            if telescope:
                return self.so_obs_paths.get(telescope, "Not available")
            return self.so_obs_paths

        # If so_obs_paths is a string (legacy format)
        if telescope:
            return (
                self.so_obs_paths if telescope in self.so_obs_paths else "Not available"
            )
        return {"all": self.so_obs_paths}

    def get_class_observations(self):
        """
        Get CLASS observation paths.

        Returns
        -------
        str or list
            Path string, list of paths, or "Not available"
        """
        if not self._loaded:
            if not self.load_flight_data():
                return "Not available"

        if not self.class_obs_paths:
            return "Not available"

        return self.class_obs_paths

    def get_all_paths(self) -> Dict[str, Any]:
        """
        Get all paths as a dictionary.

        Returns
        -------
        dict
            Dictionary with all paths and metadata
        """
        if not self._loaded:
            if not self.load_flight_data():
                return {}

        return {
            "flight_name_or_path": self.flight_name_or_path,
            "campaign": self.campaign_data,
            "payload": self.payload_data,
            "flight": self.flight_data,
            "paths": {
                "drone_data_folder": self.drone_data_folder,
                "aux_data_folder": self.aux_data_folder,
                "processed_data_folder": self.processed_data_folder,
                "so_obs": self.so_obs_paths,
                "class_obs": self.class_obs_paths,
            },
        }
