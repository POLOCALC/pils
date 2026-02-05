#!/usr/bin/env python
"""
Example usage of the PILS (POLOCALC Inertial & Drone Loading System) package.

This script demonstrates the main use cases:
1. Loading flights from STOUT database
2. Accessing decoded sensor data
3. Working with multiple flights by date range
"""

import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def example_basic_flight_loading():
    """Example: Load a single flight"""
    print("\n" + "=" * 80)
    print("Example 1: Load a Single Flight")
    print("=" * 80)

    from pils import FlightDataHandler

    try:
        # Initialize handler with STOUT database
        handler = FlightDataHandler(use_stout=True)

        # Get available campaigns
        campaigns = handler.get_campaigns()
        print(f"Available campaigns: {len(campaigns)}")

        if campaigns:
            campaign = campaigns[0]
            print(f"First campaign: {campaign}")

            # Load flights from first campaign
            campaign_id = campaign.get("campaign_id")
            if campaign_id:
                flights = handler.load_campaign_flights(campaign_id=campaign_id)
            else:
                flights = []

            if flights:
                print(f"Found {len(flights)} flights")

                # Load first flight
                flight = flights[0]
                print(f"\nLoaded flight: {flight.flight_name}")
                print(f"  Flight ID: {flight.flight_id}")
                print(f"  Campaign: {flight.campaign_id}")
                print(f"  Takeoff: {flight.takeoff_datetime}")

                # Check what data is available
                if flight.drone:
                    print(f"  ✓ Drone data: {flight.drone.data.shape}")
                if flight.payload:
                    print(f"  ✓ Payload available")
                    if flight.payload.gps:
                        print(f"    - GPS data: {flight.payload.gps.data.shape}")
                if flight.raw_files:
                    print(
                        f"  ✓ Raw files: {sum(len(v) for v in flight.raw_files.values())} files"
                    )

    except ImportError:
        print("STOUT not installed. Run: pip install 'polocalc-data-loader[stout]'")
    except Exception as e:
        print(f"Error: {e}")


def example_filesystem_mode():
    """Example: Use filesystem directly (no STOUT required)"""
    print("\n" + "=" * 80)
    print("Example 2: Filesystem Mode (No STOUT)")
    print("=" * 80)

    from pils import FlightDataHandler

    # Specify your data path
    data_path = "/mnt/data/POLOCALC"  # Adjust to your path

    try:
        handler = FlightDataHandler(use_stout=False, base_data_path=data_path)

        # Load all flights
        all_flights = handler.load_flights_by_date(
            start_date="2025-01-01", end_date="2025-01-31"
        )

        print(f"Found {len(all_flights)} flights in date range")

        for flight in all_flights[:3]:  # Show first 3
            print(f"\n  {flight.flight_name}")
            print(f"    Campaign: {flight.campaign_id}")
            print(f"    Path: {flight.flight_path}")

    except Exception as e:
        print(f"Note: Example requires data at {data_path}")
        print(f"Error: {e}")


def example_data_decoder():
    """Example: Use DataDecoder directly"""
    print("\n" + "=" * 80)
    print("Example 3: Direct DataDecoder Usage")
    print("=" * 80)

    from pils import DataDecoder

    flight_path = "/path/to/flight_20250115_1430"

    try:
        decoder = DataDecoder(flight_path, drone_model="dji")

        # Get summary without loading all data
        summary = decoder.get_data_summary()
        print(f"Flight data summary:")
        print(f"  Has drone data: {summary['drone_folder']}")
        print(f"  Has aux data: {summary['aux_folder']}")
        print(f"  Has processed data: {summary['proc_folder']}")
        print(f"  Available sensors: {summary['available_sensors']}")

        # Load specific sensors
        if summary["available_sensors"]:
            sensors = decoder.load_specific_sensors(["gps", "imu"])
            print(f"  Loaded {len(sensors)} sensor types")

    except Exception as e:
        print(f"Note: Example requires flight data at {flight_path}")
        print(f"Error: {e}")


def example_date_range_queries():
    """Example: Load flights by date range"""
    print("\n" + "=" * 80)
    print("Example 4: Date Range Queries")
    print("=" * 80)

    from pils import FlightDataHandler

    try:
        handler = FlightDataHandler(use_stout=True)

        # Load flights from a date range
        flights = handler.load_flights_by_date(
            start_date="2025-01-10",
            end_date="2025-01-20",
            decode=False,  # Don't decode, just get metadata
        )

        print(f"Flights between 2025-01-10 and 2025-01-20:")
        for flight in flights:
            print(f"  - {flight.flight_name}")
            print(f"    Campaign: {flight.campaign_id}")
            print(f"    Takeoff: {flight.takeoff_datetime}")

    except Exception as e:
        print(f"Error: {e}")


def example_stout_data_loader():
    """Example: Use StoutDataLoader directly"""
    print("\n" + "=" * 80)
    print("Example 5: StoutDataLoader (Low-Level)")
    print("=" * 80)

    from pils import StoutDataLoader

    try:
        loader = StoutDataLoader(use_stout=True)

        # Get all campaigns
        campaigns = loader.get_campaign_list()
        print(f"Available campaigns: {len(campaigns)}")

        # Get all flights
        all_flights = loader.load_all_campaign_flights()
        print(f"Total flights: {len(all_flights)}")

        # Access specific data
        if all_flights:
            flight_meta = all_flights[0]
            print(f"\nFirst flight metadata:")
            for key, value in flight_meta.items():
                if not key.startswith("_"):
                    print(f"  {key}: {value}")

    except Exception as e:
        print(f"Error: {e}")


def example_load_single_flight_payload():
    """Example: Load all payload data from a single flight"""
    print("\n" + "=" * 80)
    print("Example 6: Load Single Flight Payload Data")
    print("=" * 80)

    from pils.datahandler import Payload

    # Path to flight aux folder
    aux_path = "/mnt/data/POLOCALC/campaigns/202511/20251201/flight_20251201_1515/aux"

    try:
        # Initialize and load all sensors
        print(f"Loading payload from: {aux_path}")
        payload = Payload(dirpath=aux_path)

        # =================================================================
        # Show detected sensors from config.yml
        # =================================================================
        print("\n--- Sensors Detected from config.yml ---")
        available = payload.get_available_sensors()
        print(f"Available sensors (types): {available}")

        # Get detailed availability including file status
        sensor_availability = payload.get_sensor_availability()
        print(f"\n--- Sensor Availability (Config vs Files) ---")
        for sensor_name, info in sensor_availability.items():
            status = "✓" if info["files_exist"] else "✗"
            config_status = "✓" if info["configured"] else "✗"
            file_count_str = (
                f"({info['file_count']} files)" if info["is_multi_file"] else ""
            )
            print(
                f"{status} {info['friendly_name']:20} | Config: {config_status} | Files: {status} {file_count_str}"
            )
            if info["file_paths"]:
                for file_path in info["file_paths"]:
                    print(f"  └─ {file_path}")

        if payload.config:
            print(f"\n--- Hardware Configuration ---")
            sensors_config = payload.config.get("sensors", {})
            for sensor_name, sensor_info in sensors_config.items():
                sensor_type = sensor_info.get("sensor_info", {}).get("type", "Unknown")
                manufacturer = sensor_info.get("sensor_info", {}).get(
                    "manufacturer", "Unknown"
                )
                name = sensor_info.get("name", "Unknown")
                print(f"  {sensor_name}: {name} ({sensor_type}) - {manufacturer}")

            if payload.config.get("camera"):
                cam = payload.config["camera"]
                print(
                    f"  Camera: {cam.get('name', 'Unknown')} - Mode: {cam.get('mode', 'Unknown')}"
                )

        # =================================================================
        # Load all sensor data
        # =================================================================
        print("\n--- Loading Sensor Data ---")
        payload.load_all()

        # =================================================================
        # Access individual sensor data
        # =================================================================
        print("\n--- Loaded Sensors ---")

        # GPS data
        if payload.gps and payload.gps.data is not None:
            gps_df = payload.gps.data
            print(f"GPS: {gps_df.shape[0]} samples")
            print(f"  Columns: {list(gps_df.columns)}")
            timestamps = gps_df["timestamp"].to_numpy()
            t_start = float(timestamps[0])
            t_end = float(timestamps[-1])
            print(
                f"  Time range: {t_start:.1f}s - {t_end:.1f}s ({t_end - t_start:.1f}s duration)"
            )

        # ADC data (analog sensors)
        if payload.adc and payload.adc.data is not None:
            adc_df = payload.adc.data
            print(f"ADC: {adc_df.shape[0]} samples")
            print(f"  Columns: {list(adc_df.columns)}")

        # Inclinometer data (pitch, roll, yaw)
        if payload.inclinometer and payload.inclinometer.data is not None:
            inc_df = payload.inclinometer.data
            print(f"Inclinometer: {inc_df.shape[0]} samples")
            print(f"  Columns: {list(inc_df.columns)}")

        # IMU data (barometer, accelerometer, gyroscope, magnetometer)
        if payload.imu:
            if payload.imu.barometer and payload.imu.barometer.data is not None:
                baro_df = payload.imu.barometer.data
                print(f"Barometer: {baro_df.shape[0]} samples")
                print(f"  Columns: {list(baro_df.columns)}")

            if payload.imu.accelerometer and payload.imu.accelerometer.data is not None:
                accel_df = payload.imu.accelerometer.data
                print(f"Accelerometer: {accel_df.shape[0]} samples")
                print(f"  Columns: {list(accel_df.columns)}")

            if payload.imu.gyroscope and payload.imu.gyroscope.data is not None:
                gyro_df = payload.imu.gyroscope.data
                print(f"Gyroscope: {gyro_df.shape[0]} samples")
                print(f"  Columns: {list(gyro_df.columns)}")

            if payload.imu.magnetometer and payload.imu.magnetometer.data is not None:
                mag_df = payload.imu.magnetometer.data
                print(f"Magnetometer: {mag_df.shape[0]} samples")
                print(f"  Columns: {list(mag_df.columns)}")

        # =================================================================
        # Get sensor summary
        # =================================================================
        print("\n--- Sensor Summary ---")
        print(payload.summary())

        # =================================================================
        # Get sensor info (sample rates, time ranges)
        # =================================================================
        print("\n--- Sensor Info ---")
        info = payload.get_sensor_info()
        for sensor_name, details in info.items():
            print(f"{sensor_name}:")
            print(f"  Samples: {details['samples']}")
            print(f"  Duration: {details['duration_s']:.2f}s")
            print(f"  Sample rate: {details['sample_rate_hz']:.2f} Hz")
            print(f"  Time range: {details['t_start']:.2f}s - {details['t_end']:.2f}s")

        # =================================================================
        # Synchronize all sensors to a common time base
        # =================================================================
        print("\n--- Synchronization ---")
        max_rate = payload.get_max_sample_rate()
        print(f"Max sensor sample rate: {max_rate:.1f} Hz")

        common_range = payload.get_common_time_range()
        if common_range:
            print(f"Common time range: {common_range[0]:.2f}s - {common_range[1]:.2f}s")

            # Synchronize to 10 Hz using the Synchronizer class
            from pils.synchronizer import Synchronizer

            target_rate = min(10.0, max_rate)
            print(f"Synchronizing to {target_rate:.1f} Hz...")

            # Create synchronizer and add payload data
            sync = Synchronizer()
            if payload.gps and payload.gps.data is not None:
                sync.add_source(
                    "payload",
                    payload.gps.data.select(["timestamp", "lon", "lat", "height"]),
                    timestamp_col="timestamp",
                )

            sync_df = sync.synchronize(target_rate_hz=target_rate)
            print(
                f"Synchronized data: {sync_df.shape[0]} samples x {sync_df.shape[1]} columns"
            )
            print(f"Columns: {list(sync_df.columns)}")

            # Save to file (optional)
            # sync.save_synchronized('/tmp/synchronized_payload.parquet', format='parquet')
            # print("Saved synchronized data to /tmp/synchronized_payload.parquet")

    except FileNotFoundError:
        print(f"Note: Example requires flight data at {aux_path}")
        print("Adjust the aux_path variable to point to your data.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def example_synchronize_payload_and_drone_data():
    """Example: Synchronize payload and drone data to a common time base"""
    print("\n" + "=" * 80)
    print("Example 7: Synchronize Payload and Drone Data")
    print("=" * 80)

    from pils.datahandler import Payload
    from pils.synchronizer import Synchronizer
    import polars as pl

    aux_path = "/mnt/data/POLOCALC/campaigns/202511/20251201/flight_20251201_1515/aux"

    try:
        print("\n1. Loading Payload Data")
        print("-" * 40)
        payload = Payload(dirpath=aux_path)
        payload.load_all()
        print(f"Loaded sensors: {payload.get_available_sensors()}")

        # Get payload data as Polars DataFrames
        print("\n2. Extracting Payload Data for Synchronization")
        print("-" * 40)

        # Combine payload sensor data into a single DataFrame for synchronization
        # For this example, we'll extract GPS data with key columns
        payload_data = None
        if payload.gps is not None and payload.gps.data is not None:
            payload_data = payload.gps.data.select(
                ["timestamp", "lon", "lat", "height"]
            )
            payload_data = payload_data.rename(
                {
                    "lon": "gps_lon",
                    "lat": "gps_lat",
                    "height": "gps_height",
                }
            )
            print(f"Payload GPS shape: {payload_data.shape}")
            print(f"Columns: {payload_data.columns}")

        # Simulate drone data (in real scenario, load from drone data source)
        print("\n3. Creating Sample Drone Data")
        print("-" * 40)
        import numpy as np

        # Create synthetic drone data with overlapping time range
        if payload_data is not None:
            timestamps = payload_data["timestamp"].to_numpy()
            t_start = timestamps[0]
            t_end = timestamps[-1]

            # Drone data at 10 Hz
            drone_timestamps = np.arange(t_start, t_end, 0.1)  # 10 Hz sampling rate
            n_drone_samples = len(drone_timestamps)

            drone_data = pl.DataFrame(
                {
                    "timestamp": drone_timestamps,
                    "drone_x": np.random.randn(n_drone_samples).cumsum() * 10,
                    "drone_y": np.random.randn(n_drone_samples).cumsum() * 10,
                    "drone_z": np.random.randn(n_drone_samples).cumsum() * 5 + 100,
                    "drone_yaw": np.random.randn(n_drone_samples) * 180,
                }
            )
            print(f"Drone data shape: {drone_data.shape}")
            print(f"Columns: {drone_data.columns}")

            # Now synchronize payload and drone data
            print("\n4. Synchronizing Payload and Drone Data")
            print("-" * 40)

            sync = Synchronizer()

            # Add payload data source
            sync.add_source("payload", payload_data, timestamp_col="timestamp")

            # Add drone data source
            sync.add_source("drone", drone_data, timestamp_col="timestamp")

            # Get source information
            print("\nData source information:")
            source_info = sync.get_source_info()
            for source_name, info in source_info.items():
                print(f"\n  {source_name.upper()}:")
                print(f"    Samples: {info['samples']}")
                print(f"    Duration: {info['duration_s']:.2f} s")
                print(f"    Sample rate: {info['sample_rate_hz']:.2f} Hz")
                print(f"    Time range: {info['t_start']:.3f} - {info['t_end']:.3f} s")

            # Perform synchronization
            print("\n5. Creating Synchronized Dataset")
            print("-" * 40)

            # Synchronize at 2 Hz (the GPS sample rate)
            synchronized = sync.synchronize(target_rate_hz=2.0)

            print(f"Synchronized data shape: {synchronized.shape}")
            print(f"Synchronized data columns: {synchronized.columns}")
            print(f"\nFirst 5 rows of synchronized data:")
            print(synchronized.head())

            # Save synchronized data
            print("\n6. Saving Synchronized Data")
            print("-" * 40)
            output_path = "/tmp/synchronized_payload_drone.parquet"
            sync.save_synchronized(output_path, format="parquet")
            print(f"Saved to: {output_path}")

            print("\n" + "=" * 80)
            print("Example 7 completed successfully!")
            print("=" * 80)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


def main():
    """Run all examples"""
    print("\nPILS (POLOCALC Inertial & Drone Loading System) - Usage Examples")
    print("================================================================\n")

    # Run examples
    # example_basic_flight_loading()
    # example_filesystem_mode()
    # example_data_decoder()
    # example_date_range_queries()
    # example_stout_data_loader()
    # example_load_single_flight_payload()
    example_synchronize_payload_and_drone_data()

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
