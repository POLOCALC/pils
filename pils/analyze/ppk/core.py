"""PPKAnalysis: RTKLIB post-processing of a flight's GPS, with versioned results."""

from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import polars as pl

from pils.analyze.ppk.inputs import PPKInputsMixin
from pils.analyze.ppk.storage import PPKStorageMixin
from pils.analyze.ppk.versions import PPKVersion
from pils.analyze.ppk_tuning import rtklib_binary, tune
from pils.analyze.ppkdata.PPK.pos_analyzer import POSAnalyzer
from pils.analyze.ppkdata.PPK.report import RTKLIBReport
from pils.analyze.ppkdata.PPK.stat_analyzer import STATAnalyzer
from pils.analyze.ppkdata.RINEX.report import RINEXReport
from pils.flight import Flight
from pils.integrations.stout import layout

logger = logging.getLogger(__name__)


class PPKAnalysis(PPKInputsMixin, PPKStorageMixin):
    """
    Standalone PPK analysis manager with smart execution and versioning.

    Manages RTKLIB-based post-processing of GPS data with intelligent
    re-run logic, automatic versioning, and HDF5 persistence. Completely
    separate from Flight class.

    The analysis system:
    - Only runs RTKLIB when config changes (SHA256 hash comparison)
    - Stores each revision in its own folder with .pos, .stat, .conf files
    - Saves all versions to a single ppk_solution.h5 HDF5 file
    - Generates auto-timestamp version names

    File Structure
    --------------
    flight_dir/proc/ppk/
    ├── ppk_solution.h5              # HDF5 with all versions
    └── rev_20260204_143022/         # Per-revision folder
        ├── solution.pos              # RTKLIB position output
        ├── solution.pos.stat         # RTKLIB statistics output
        └── config.conf               # RTKLIB config used

    Attributes
    ----------
    flight_path : Path
        Root flight directory
    ppk_dir : Path
        PPK directory ({flight_path}/proc/ppk/)
    hdf5_path : Path
        Path to ppk_solution.h5 file
    versions : Dict[str, PPKVersion]
        Dictionary of loaded PPK versions

    Examples
    --------
    >>> from pils.flight import Flight
    >>> # Create Flight object
    >>> flight_info = {
    ...     "drone_data_folder_path": "/path/to/flight/drone",
    ...     "aux_data_folder_path": "/path/to/flight/aux"
    ... }
    >>> flight = Flight(flight_info)
    >>> # Create new analysis
    >>> ppk = PPKAnalysis(flight)
    >>> version = ppk.run_analysis('rtklib.conf')  # Smart execution
    >>> # Only runs if config changed
    >>> version2 = ppk.run_analysis('rtklib.conf')  # Skipped if same
    >>> # Force re-run
    >>> version3 = ppk.run_analysis('rtklib.conf', force=True)
    >>> # Access versions
    >>> latest = ppk.get_latest_version()
    >>> all_versions = ppk.list_versions()
    """

    def __init__(self, flight: Flight):
        """
        Initialize PPKAnalysis for a flight.

        Creates the proc/ppk directory structure if it doesn't exist.

        Parameters
        ----------
        flight : Flight
            Flight object with valid flight_path attribute

        Raises
        ------
        TypeError
            If flight is not a Flight object
        ValueError
            If flight.flight_path is None or not an existing directory

        Examples
        --------
        >>> from pils.flight import Flight
        >>> flight_info = {
        ...     "drone_data_folder_path": "/path/to/flight/drone",
        ...     "aux_data_folder_path": "/path/to/flight/aux"
        ... }
        >>> flight = Flight(flight_info)
        >>> ppk = PPKAnalysis(flight)
        >>> print(ppk.ppk_dir)
        /path/to/flight/proc/ppk
        """
        # Validate input type
        if not isinstance(flight, Flight):
            raise TypeError(
                f"Expected Flight object, got {type(flight).__name__}. "
                "PPKAnalysis now requires a Flight object instead of a path."
            )

        # Validate flight_path exists
        if flight.flight_path is None:
            raise ValueError(
                "Flight object must have a valid flight_path attribute. "
                "Ensure the Flight was initialized with proper flight_info."
            )

        # Convert to Path and validate it's a directory
        flight_path = Path(flight.flight_path)
        if not flight_path.exists() or not flight_path.is_dir():
            raise ValueError(
                f"flight_path must be an existing directory. Got: {flight_path}"
            )

        self.flight = flight
        self.flight_path = flight_path
        self.ppk_dir = self.flight_path / layout.PROC_FOLDER / "ppk"
        self.hdf5_path = self.ppk_dir / "ppk_solution.h5"
        self.versions: dict[str, PPKVersion] = {}

        # Create PPK directory structure
        self.ppk_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized PPKAnalysis for flight: {self.flight_path}")
        logger.info(f"PPK directory: {self.ppk_dir}")

    def run_analysis(
        self,
        config_path: str | Path,
        rover_obs: str | Path | None = None,
        base_obs: str | Path | None = None,
        nav_file: str | Path | None = None,
        force: bool = False,
        analyze_rinex: bool = False,
        analyze_ppk: bool = False,
        optimize: bool = True,
        reference: pl.DataFrame | str | None = "drone",
        selection: str = "fix_rate",
    ) -> PPKVersion | None:
        """
        Execute RTKLIB PPK analysis with smart re-run logic.

        Smart execution:
        - Runs only if config changed (hash comparison) or force=True
        - Generates auto-timestamp version name
        - Creates revision folder with .pos, .stat, .conf files
        - Executes RTKLIB rnx2rtkp subprocess
        - Parses results using POSAnalyzer and STATAnalyzer
        - Saves to HDF5

        Parameters
        ----------
        config_path : Union[str, Path]
            Path to RTKLIB configuration file
        rover_obs : Union[str, Path], optional
            Path to rover RINEX observation file. If None, first look for available
            files in the folder otherwise use convbin for conversion (default: None)
        base_obs : Union[str, Path], optional
            Path to base RINEX observation file. If None, first look for available
            files in the folder otherwise use convbin for conversion (default: None)
        nav_file : Union[str, Path], optional
            Path to navigation file. If None, first look for available
            files in the folder otherwise use convbin for conversion (default: None)
        force : bool, optional
            If True, force re-run even if config unchanged (default: False)
        analyze_rinex : bool, optional
            If True, analyze RINEX files (default: False)
        analyze_ppk : bool, optional
            If True, analyze PPK results (default: False)
        optimize : bool, default=True
            Tune the configuration to the data first (see
            :func:`pils.analyze.ppk_tuning.tune`): ``config_path`` is the
            template, the chosen configuration is saved in the revision as
            ``optimized.conf`` and every candidate in the ``tuning`` table.
            If no candidate passes validation, the template is used.
        reference : pl.DataFrame, "drone" or None, optional
            Track for ``selection="reference"``: ``"drone"`` (the drone's
            RTK track) or a table with ``timestamp``, ``lat_deg``,
            ``lon_deg``, ``h_ell_m``. Not loaded with the other selections.
        selection : {"fix_rate", "reference", "directions"}, default="fix_rate"
            How the tuning chooses the configuration: the most epochs fixed
            by RTKLIB (default), the most fixes agreeing with ``reference``,
            or forward/backward agreement (see
            :func:`pils.analyze.ppk_tuning.tune`).

        Returns
        -------
        Optional[PPKVersion]
            PPKVersion object with results, or None if execution failed

        Raises
        ------
        FileNotFoundError
            If config file doesn't exist

        Examples
        --------
        >>> ppk = PPKAnalysis('/path/to/flight')
        >>> # Smart execution - only runs if needed
        >>> v1 = ppk.run_analysis(
        ...     'rtklib.conf',
        ...     'rover.obs',
        ...     'base.obs',
        ...     'nav.nav'
        ... )
        >>> v2 = ppk.run_analysis(...)  # Skipped if config same
        >>> # Force re-run
        >>> v3 = ppk.run_analysis(..., force=True)
        """
        before = {p.name for p in self.ppk_dir.glob(layout.REVISION_PREFIX + "*")}
        try:
            return self._run_analysis(
                config_path,
                rover_obs=rover_obs,
                base_obs=base_obs,
                nav_file=nav_file,
                force=force,
                analyze_rinex=analyze_rinex,
                analyze_ppk=analyze_ppk,
                optimize=optimize,
                reference=reference,
                selection=selection,
            )
        finally:
            # A run that failed leaves its revision folder without a manifest
            for folder in self.ppk_dir.glob(layout.REVISION_PREFIX + "*"):
                if (
                    folder.name not in before
                    and not (folder / "manifest.json").exists()
                ):
                    shutil.rmtree(folder, ignore_errors=True)

    def _run_analysis(
        self,
        config_path: str | Path,
        rover_obs: str | Path | None = None,
        base_obs: str | Path | None = None,
        nav_file: str | Path | None = None,
        force: bool = False,
        analyze_rinex: bool = False,
        analyze_ppk: bool = False,
        optimize: bool = True,
        reference: pl.DataFrame | str | None = "drone",
        selection: str = "fix_rate",
    ) -> PPKVersion | None:
        """Run the analysis; see :meth:`run_analysis`."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Check if should run
        if not force and not self._should_run_analysis(
            config_path, optimize, selection
        ):
            logger.info("Skipping analysis - config unchanged")
            latest = self.get_latest_version()
            if latest is not None:
                return latest

        # Generate version name and create folder
        version_name = self._generate_version_name()
        revision_path = self._create_revision_folder(version_name)

        # Initialize nav file variables
        rover_nav: Path | None = None
        base_nav: Path | None = None

        if rover_obs is not None:
            rover_obs = Path(rover_obs)
            # Infer rover nav file if exists
            potential_nav = rover_obs.with_suffix(".nav")
            if potential_nav.exists():
                rover_nav = potential_nav
        else:
            obs_path = self.ppk_dir / "rover.obs"
            nav_path = self.ppk_dir / "rover.nav"

            if obs_path.exists() and nav_path.exists():
                rover_obs = obs_path
                rover_nav = nav_path

            else:
                rover_path = self.flight_path / layout.AUX_FOLDER / "sensors"

                try:
                    rover_ubx = sorted(
                        [*rover_path.glob("*_GPS.bin"), *rover_path.glob("*_GPS.ubx")]
                    )[0]
                except IndexError:
                    logger.info("Missing rover .bin file")
                    return None

                cmd = [
                    rtklib_binary("convbin"),
                    "-od",
                    "-os",
                    "-oi",
                    "-ot",
                    "-ol",
                    "-r",
                    "ubx",
                    "-o",
                    str(obs_path),
                    "-n",
                    str(nav_path),
                    str(rover_ubx),
                ]

                subprocess.run(cmd, check=True)

                if not obs_path.exists():
                    logger.info("rover obs file not created")
                    raise FileNotFoundError("Conversion failed: .obs file not created")
                if not obs_path.exists():
                    logger.info("rover nav file not created")
                    raise FileNotFoundError("Conversion failed: .nav file not created")

                rover_obs = obs_path
                rover_nav = nav_path

        start_rover, end_rover = self._get_rinex_bounds(rover_obs)

        if base_obs is not None:
            base_obs = Path(base_obs)
            # Infer base nav file if exists
            potential_nav = base_obs.with_suffix(".nav")
            if potential_nav.exists():
                base_nav = potential_nav
        else:
            obs_path = self.ppk_dir / "base.obs"
            nav_path = self.ppk_dir / "base.nav"
            if start_rover is None or end_rover is None:
                raise ValueError("Could not determine rover observation time bounds")
            cached = obs_path.exists() and nav_path.exists()
            if cached and not self._covers(obs_path, start_rover, end_rover):
                logger.info(
                    "Cached base.obs does not cover the rover: converting again"
                )
                cached = False
            if not cached:
                self._convert_base(obs_path, nav_path, start_rover, end_rover)
            base_obs = obs_path
            base_nav = nav_path

        if nav_file is not None:
            nav_file = Path(nav_file)
        else:
            # Use the larger navigation file if both exist
            if base_nav and rover_nav:
                nav_file = max(base_nav, rover_nav, key=lambda p: p.stat().st_size)
            elif base_nav:
                nav_file = base_nav
            elif rover_nav:
                nav_file = rover_nav
            else:
                raise FileNotFoundError("No navigation file available (base or rover)")

        if not rover_obs.exists():
            raise FileNotFoundError(f"Rover observation file not found: {rover_obs}")
        if not base_obs.exists():
            raise FileNotFoundError(f"Base observation file not found: {base_obs}")
        if not nav_file.exists():
            raise FileNotFoundError(f"Navigation file not found: {nav_file}")

        if analyze_rinex:
            obs_files = [rover_obs, base_obs]
            nav_files = [rover_nav, base_nav]

            names = ["rover", "base"]

            for i, obs in enumerate(obs_files):
                report_md = self.ppk_dir / f"report_{names[i]}.md"

                if report_md.exists():
                    continue
                else:
                    report = RINEXReport(obs, nav_files[i])

                    plot_folder = self.ppk_dir / f"rinex_plots_{names[i]}"

                    report.generate(str(report_md), str(plot_folder.name))

        logger.info(f"Running PPK analysis: {version_name}")
        tuning = None
        if optimize:
            track = None
            if selection == "reference":
                track = (
                    self._reference_track() if isinstance(reference, str) else reference
                )
            tuning = tune(
                config_path,
                rover_obs,
                base_obs,
                nav_file,
                revision_path / "tuning",
                reference=track,
                selection=selection,
            )
            shutil.rmtree(revision_path / "tuning", ignore_errors=True)
        if tuning is not None and tuning.config_text is not None:
            config_copy = revision_path / "optimized.conf"
            config_copy.write_text(tuning.config_text)
            logger.info(f"Tuned config saved to {config_copy}")
        else:
            if optimize:
                logger.warning(
                    "Tuning found no valid configuration: using the template"
                )
            # Copy config to revision folder
            config_copy = revision_path / config_path.name
            shutil.copy2(config_path, config_copy)
            logger.info(f"Copied config to {config_copy}")

        # Define output file paths
        pos_file = revision_path / "solution.pos"
        stat_file = revision_path / "solution.pos.stat"

        cmd = [
            rtklib_binary("rnx2rtkp"),
            "-k",
            str(config_copy),
            "-o",
            str(pos_file),
            str(rover_obs),
            str(base_obs),
            str(nav_file),
        ]

        subprocess.run(cmd, check=True)

        if analyze_ppk:
            report_md = revision_path / "report_ppk.md"

            report = RTKLIBReport(pos_file, stat_file)

            plot_folder = revision_path / "plots"

            report.generate(str(report_md.parent), str(plot_folder.name))

        # Check if output files were created
        if not pos_file.exists():
            logger.error(f"Position file not created: {pos_file}")
            # Clean up revision folder on failure
            if revision_path.exists():
                shutil.rmtree(revision_path)
                logger.info(f"Cleaned up failed revision folder: {revision_path}")
            return None

        # Parse position file
        try:
            pos_analyzer = POSAnalyzer(str(pos_file))
            pos_data = pos_analyzer.parse()
            logger.info(f"Parsed position: {len(pos_data)} epochs")
        except Exception as e:
            logger.error(f"Failed to parse position file: {e}")
            # Clean up revision folder on failure
            if revision_path.exists():
                shutil.rmtree(revision_path)
                logger.info(f"Cleaned up failed revision folder: {revision_path}")
            return None

        # Parse statistics file (optional)
        if not stat_file.exists():
            logger.warning(f"Statistics file not created: {stat_file}")
            stat_data = None
        else:
            try:
                stat_analyzer = STATAnalyzer(str(stat_file))
                stat_data = stat_analyzer.parse()
                logger.info(f"Parsed statistics: {len(stat_data)} records")
            except Exception as e:
                logger.error(f"Failed to parse statistics file: {e}")
                stat_data = None

        # Create metadata with config hash and parsed parameters
        config_hash = self._hash_config(config_path) + (
            f":optimize:{selection}" if optimize else ""
        )
        config_params = self._parse_config_params(config_copy)
        metadata = {
            "config_hash": config_hash,
            "timestamp": datetime.now().isoformat(),
            "config_params": config_params,
            "rover_obs": str(rover_obs),
            "base_obs": str(base_obs),
            "nav_file": str(nav_file),
            "pos_file": str(pos_file),
            "stat_file": str(stat_file) if stat_data is not None else None,
            "config_template": str(config_path),
            "config_used": str(config_copy),
            **(tuning.metadata() if tuning is not None else {}),
        }

        # Create version object (handle None stat_data)
        if stat_data is None:
            stat_data = pl.DataFrame()

        version = PPKVersion(
            version_name=version_name,
            pos_data=pos_data,
            stat_data=stat_data,
            metadata=metadata,
            revision_path=revision_path,
            tuning_data=tuning.table() if tuning is not None else None,
        )

        # Store in versions dict
        self.versions[version_name] = version

        # Save automatically
        self._save_version(version)

        logger.info(f"PPK analysis complete: {version_name}")
        return version
