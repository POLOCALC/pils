import pandas as pd
import numpy as np
import datetime
import scipy as sp

import os
import sys

from get_payload_data import  read_adc_file, read_gps_file, read_inclino_file
from pils.tools import get_path_from_keyword

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process data from the payload sensors.")
    parser.add_argument("dirpath", type=str, help="Directory path to the folder containing 'sensors_data' folder and 'file.log' with sensor files.")
    parser.add_argument("-g", "--gps", action="store_true", help="Process the GPS quick look")
    parser.add_argument("-k", "--kernel", action="store_true", help="Process the Kernel-100 inclinometer quick look")
    parser.add_argument("-a", "--adc", action="store_true", help="Process the ADC quick look")
    parser.add_argument("-i", "--inertial", action="store_true", help="Process the barometer, magnetometer, accelerometer and gyroscope quick look")
    parser.add_argument("-o", "--output-dir", default="", help="Output directory to save the csv files.")


    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = f"./process_data/{os.path.basename(os.path.normpath(args.dirpath))}"

    output_dir = os.path.join(args.output_dir, "csv_outputs")
    print("All plots saved in : ", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    logfile = get_path_from_keyword(args.dirpath, "file.log")


    print(f"All csv files saved in {output_dir}")

    #Process all of them if no arguments given, only the one asked if the argument are given
    if not args.gps and not args.kernel and not args.adc and not args.inertial:
         args.gps, args.kernel, args.adc, args.inertial = True, True, True, True
         
    if args.inertial:
        barometer_path = get_path_from_keyword(args.dirpath, "barometer.bin")
        if barometer_path is not None:
            baro = pd.read_csv(barometer_path, delimiter=' ', names=["timestamp", "pressure", "temperature"])
            baro.to_csv(os.path.join(output_dir, "barometer.csv"))
            print("Barometer file saved")

        magnetometer_path = get_path_from_keyword(args.dirpath, "magnetometer.bin")
        if magnetometer_path is not None:
            magneto = pd.read_csv(magnetometer_path, delimiter=' ', names=["timestamp", "mag_x", "mag_y", "mag_z"])
            magneto.to_csv(os.path.join(output_dir, "magnetometer.csv"))
            print("Magnetometer file saved")

        accelerometer_path = get_path_from_keyword(args.dirpath, "accelerometer.bin")
        if accelerometer_path is not None:
            accelero = pd.read_csv(accelerometer_path, delimiter=' ', names=["timestamp", "acc_x", "acc_y", "acc_z"])
            accelero.to_csv(os.path.join(output_dir, "accelerometer.csv"))
            print("Accelerometer file saved")

        gyroscope_path = get_path_from_keyword(args.dirpath, "gyroscope.bin")
        if gyroscope_path is not None:
            gyro = pd.read_csv(gyroscope_path, delimiter=' ', names=["timestamp", "x", "y", "z"])
            gyro.to_csv(os.path.join(output_dir, "gyroscope.csv"))
            print("Gyroscope file saved")

    if args.gps:
        gps_path = get_path_from_keyword(args.dirpath, "ZED-F9P")
        if gps_path is not None:
            gps_data = read_gps_file(gps_path=gps_path, logfile=logfile)
            gps_data.to_csv(os.path.join(output_dir, "gps.csv"))
            print("GPS file saved")

    if args.adc:
        adc_path = get_path_from_keyword(args.dirpath, "ADS1015")
        if adc_path is not None:
            adc_data = read_adc_file(adc_path=adc_path)
            adc_data.to_csv(os.path.join(output_dir, "adc.csv"))
            print("ADC file saved")

    if args.kernel:
        inclino_path = get_path_from_keyword(args.dirpath, "Kernel-100")
        if inclino_path is not None:
            inclino_data = read_inclino_file(inclino_path=inclino_path, logfile=logfile)
            inclino_data.to_csv(os.path.join(output_dir, "inclinometer.csv"))
            print("Inclinometer file saved")




