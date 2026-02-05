"""
PPK Analysis Example - Complete workflow for RTKLIB-based GPS post-processing.

This example demonstrates the full PPK analysis workflow using PILS:
1. Load flight data
2. Initialize PPK analysis
3. Run RTKLIB processing with smart execution
4. Access results and manage versions
5. Load from HDF5 for later analysis

Requirements:
- RTKLIB rnx2rtkp binary in PATH (or specify full path)
- RINEX observation files (rover and base)
- RINEX navigation file
- RTKLIB configuration file
"""

from pathlib import Path
from pils.analyze.ppk import PPKAnalysis

# =============================================================================
# Example 1: Basic PPK Analysis
# =============================================================================


def example_basic_ppk_analysis():
    """Run PPK analysis with smart execution."""

    # Paths to input files
    flight_path = Path("/path/to/flight_20260204_1430")
    config_file = Path("/path/to/rtklib.conf")
    rover_obs = Path("/path/to/rover.obs")
    base_obs = Path("/path/to/base.obs")
    nav_file = Path("/path/to/nav.nav")

    # Initialize PPK analysis
    # Creates proc/ppk/ directory and ppk_solution.h5 file
    ppk = PPKAnalysis(flight_path)

    # Run analysis - smart execution only runs if config changed
    version = ppk.run_analysis(
        config_path=config_file,
        rover_obs=rover_obs,
        base_obs=base_obs,
        nav_file=nav_file,
        force=False,  # Set to True to force re-run
    )

    if version:
        print(f"Analysis completed: {version.version_name}")
        print(f"Position epochs: {len(version.pos_data)}")
        print(f"Statistics records: {len(version.stat_data) if version.stat_data else 0}")

        # Access position data (Polars DataFrame)
        print("\nPosition data columns:")
        print(version.pos_data.columns)

        # Access statistics data (Polars DataFrame)
        if version.stat_data:
            print("\nStatistics data columns:")
            print(version.stat_data.columns)

        # Access metadata
        print("\nMetadata:")
        print(f"  Config hash: {version.metadata['config_hash']}")
        print(f"  Timestamp: {version.metadata['timestamp']}")
        print(f"  Rover: {version.metadata['rover_obs']}")
        print(f"  Base: {version.metadata['base_obs']}")


# =============================================================================
# Example 2: Smart Execution - Config Change Detection
# =============================================================================


def example_smart_execution():
    """Demonstrate smart execution with config change detection."""

    flight_path = Path("/path/to/flight_20260204_1430")
    config_file = Path("/path/to/rtklib.conf")
    rover_obs = Path("/path/to/rover.obs")
    base_obs = Path("/path/to/base.obs")
    nav_file = Path("/path/to/nav.nav")

    ppk = PPKAnalysis(flight_path)

    # First run - executes RTKLIB
    print("First run:")
    v1 = ppk.run_analysis(config_file, rover_obs, base_obs, nav_file)
    print(f"  Version: {v1.version_name}")

    # Second run with same config - skips execution
    print("\nSecond run (same config):")
    v2 = ppk.run_analysis(config_file, rover_obs, base_obs, nav_file)
    print(f"  Version: {v2.version_name}")
    print(f"  Same version? {v1.version_name == v2.version_name}")  # True

    # Modify config and run again - executes RTKLIB
    # (modify rtklib.conf file here)

    # Third run with changed config - executes RTKLIB
    print("\nThird run (changed config):")
    v3 = ppk.run_analysis(config_file, rover_obs, base_obs, nav_file)
    print(f"  Version: {v3.version_name}")
    print(f"  Same as v1? {v1.version_name == v3.version_name}")  # False

    # Force re-run even if config unchanged
    print("\nForced run:")
    v4 = ppk.run_analysis(config_file, rover_obs, base_obs, nav_file, force=True)
    print(f"  Version: {v4.version_name}")


# =============================================================================
# Example 3: Version Management
# =============================================================================


def example_version_management():
    """Manage multiple PPK analysis versions."""

    flight_path = Path("/path/to/flight_20260204_1430")

    # Load existing analysis from HDF5
    ppk = PPKAnalysis.from_hdf5(flight_path)

    # List all versions
    versions = ppk.list_versions()
    print(f"Total versions: {len(versions)}")
    for v_name in versions:
        print(f"  - {v_name}")

    # Get latest version
    latest = ppk.get_latest_version()
    if latest:
        print(f"\nLatest version: {latest.version_name}")
        print(f"  Config: {latest.metadata.get('config_params', {}).get('pos1-posmode')}")
        print(f"  Epochs: {len(latest.pos_data)}")

    # Get specific version by name
    specific = ppk.get_version("rev_20260204_143022")
    if specific:
        print(f"\nSpecific version: {specific.version_name}")
        print(f"  Position data shape: {specific.pos_data.shape}")

    # Delete old version
    ppk.delete_version("rev_20260203_120000")
    print(f"\nVersions after deletion: {len(ppk.list_versions())}")


# =============================================================================
# Example 4: Analyzing Position Quality
# =============================================================================


def example_position_quality_analysis():
    """Analyze PPK position solution quality."""

    flight_path = Path("/path/to/flight_20260204_1430")

    ppk = PPKAnalysis.from_hdf5(flight_path)
    latest = ppk.get_latest_version()

    if not latest:
        print("No PPK versions found")
        return

    pos_df = latest.pos_data

    # Filter by solution quality (Q=1: Fixed, Q=2: Float, Q=5: Single)
    fixed_solutions = pos_df.filter(pl.col("Q") == 1)
    float_solutions = pos_df.filter(pl.col("Q") == 2)

    print("Position Quality Summary:")
    print(f"  Total epochs: {len(pos_df)}")
    print(
        f"  Fixed solutions: {len(fixed_solutions)} ({len(fixed_solutions)/len(pos_df)*100:.1f}%)"
    )
    print(
        f"  Float solutions: {len(float_solutions)} ({len(float_solutions)/len(pos_df)*100:.1f}%)"
    )

    # Compute statistics for fixed solutions
    if len(fixed_solutions) > 0:
        import numpy as np

        print("\nFixed Solution Statistics:")
        print(f"  Mean sdn: {fixed_solutions['sdn'].mean():.4f} m")
        print(f"  Mean sde: {fixed_solutions['sde'].mean():.4f} m")
        print(f"  Mean sdu: {fixed_solutions['sdu'].mean():.4f} m")
        print(f"  Mean satellites: {fixed_solutions['ns'].mean():.1f}")

        # 3D position uncertainty
        pos_3d_std = np.sqrt(
            fixed_solutions["sdn"].to_numpy() ** 2
            + fixed_solutions["sde"].to_numpy() ** 2
            + fixed_solutions["sdu"].to_numpy() ** 2
        )
        print(f"  Mean 3D uncertainty: {pos_3d_std.mean():.4f} m")


# =============================================================================
# Example 5: Complete Flight Workflow with PPK
# =============================================================================


def example_complete_workflow():
    """Complete workflow: Flight data + PPK analysis."""

    from pils import Flight

    # Step 1: Load flight data
    flight_info = {
        "drone_data_folder_path": "/path/to/flight/drone",
        "aux_data_folder_path": "/path/to/flight/aux",
    }
    flight = Flight(flight_info)
    flight.add_drone_data()
    flight.add_sensor_data(["gps", "imu"])

    print("Flight data loaded:")
    print(f"  Drone data: {flight.raw_data.drone_data.drone.shape}")
    print(f"  GPS data: {flight.raw_data.payload_data.gps.shape}")

    # Step 2: Run PPK analysis (completely separate from Flight)
    flight_path = Path(flight_info["drone_data_folder_path"]).parent

    ppk = PPKAnalysis(flight_path)
    version = ppk.run_analysis(
        config_path="/path/to/rtklib.conf",
        rover_obs="/path/to/rover.obs",
        base_obs="/path/to/base.obs",
        nav_file="/path/to/nav.nav",
    )

    print("\nPPK analysis completed:")
    print(f"  Version: {version.version_name}")
    print(f"  Position epochs: {len(version.pos_data)}")

    # Step 3: Compare PPK with payload GPS
    import polars as pl

    payload_gps = flight.raw_data.payload_data.gps
    ppk_pos = version.pos_data

    print("\nData comparison:")
    print(f"  Payload GPS epochs: {len(payload_gps)}")
    print(f"  PPK epochs: {len(ppk_pos)}")
    print(
        f"  Time overlap: {max(payload_gps['timestamp'].min(), ppk_pos['timestamp'].min())} to {min(payload_gps['timestamp'].max(), ppk_pos['timestamp'].max())}"
    )


# =============================================================================
# Example 6: Working with Multiple Flights
# =============================================================================


def example_batch_processing():
    """Process multiple flights in batch."""

    from pathlib import Path

    # List of flights to process
    flights = [
        Path("/path/to/flight_20260204_1430"),
        Path("/path/to/flight_20260204_1500"),
        Path("/path/to/flight_20260204_1530"),
    ]

    config_file = Path("/path/to/rtklib.conf")

    results = []

    for flight_path in flights:
        print(f"\nProcessing {flight_path.name}...")

        # Find RINEX files in aux folder
        aux_path = flight_path / "aux"
        rover_obs = list(aux_path.glob("*_rover.obs"))[0]
        base_obs = list(aux_path.glob("*_base.obs"))[0]
        nav_file = list(aux_path.glob("*_base.nav"))[0]

        # Run PPK analysis
        ppk = PPKAnalysis(flight_path)
        version = ppk.run_analysis(
            config_path=config_file, rover_obs=rover_obs, base_obs=base_obs, nav_file=nav_file
        )

        if version:
            # Collect quality metrics
            fixed_ratio = len(version.pos_data.filter(pl.col("Q") == 1)) / len(version.pos_data)

            results.append(
                {
                    "flight": flight_path.name,
                    "version": version.version_name,
                    "epochs": len(version.pos_data),
                    "fixed_ratio": fixed_ratio,
                }
            )

            print(f"  Completed: {version.version_name}")
            print(f"  Fixed solutions: {fixed_ratio*100:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("Batch Processing Summary:")
    print("=" * 60)
    for result in results:
        print(
            f"{result['flight']:30s} {result['epochs']:6d} epochs  {result['fixed_ratio']*100:5.1f}% fixed"
        )


# =============================================================================
# Example 7: Custom RTKLIB Binary Path
# =============================================================================


def example_custom_binary_path():
    """Specify custom path to RTKLIB binary."""

    flight_path = Path("/path/to/flight_20260204_1430")
    config_file = Path("/path/to/rtklib.conf")
    rover_obs = Path("/path/to/rover.obs")
    base_obs = Path("/path/to/base.obs")
    nav_file = Path("/path/to/nav.nav")

    ppk = PPKAnalysis(flight_path)

    # Specify custom binary path (if not in PATH)
    version = ppk.run_analysis(
        config_path=config_file,
        rover_obs=rover_obs,
        base_obs=base_obs,
        nav_file=nav_file,
        rnx2rtkp_path="/usr/local/bin/rnx2rtkp",  # Custom path
    )

    print(f"Analysis completed with custom binary: {version.version_name}")


if __name__ == "__main__":
    print("PPK Analysis Examples")
    print("=" * 60)
    print("\nUncomment the example you want to run:\n")

    # Uncomment one example to run:
    # example_basic_ppk_analysis()
    # example_smart_execution()
    # example_version_management()
    # example_position_quality_analysis()
    # example_complete_workflow()
    # example_batch_processing()
    # example_custom_binary_path()
