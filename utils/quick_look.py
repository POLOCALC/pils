import argparse
import os

import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

from get_payload_data import read_inclino_file, read_gps_file, read_adc_file
from pils.tools import get_path_from_keyword
from matplotlib.backends.backend_pdf import PdfPages


##################################INERTIAL INSTRUMENTS########################################################

            
def plot_inertial(dirpath, plot_dirpath, pdf):
    barometer_path = get_path_from_keyword(dirpath, "barometer.bin")
    magnetometer_path = get_path_from_keyword(dirpath, "magnetometer.bin")
    accelerometer_path = get_path_from_keyword(dirpath, "accelerometer.bin")
    gyroscope_path = get_path_from_keyword(dirpath, "gyroscope.bin")

    with open(accelerometer_path, "r") as f:
        accel_lines = f.readlines()
    with open(gyroscope_path, "r") as f:
        gyro_lines = f.readlines()
    with open(barometer_path, "r") as f:
        bar_lines = f.readlines()
    with open(magnetometer_path, "r") as f:
        mag_lines = f.readlines()

    accel_lines = accel_lines[:-1]
    gyro_lines = gyro_lines[:-1]
    bar_lines = bar_lines[:-1]
    mag_lines = mag_lines[:-1]

    accel_data = np.loadtxt(StringIO("".join(accel_lines)), delimiter=" ")
    gyro_data = np.loadtxt(StringIO("".join(gyro_lines)), delimiter=" ")
    bar_data = np.loadtxt(StringIO("".join(bar_lines)), delimiter=" ")
    mag_data = np.loadtxt(StringIO("".join(mag_lines)), delimiter=" ")

    print("All intertial data loaded successfully")

    titles = ["Accelerometer", "Gyroscope", "Magnetometer"]
    ylabels = ["Acceleration [g]", "Angular Velocity [rad/s]", "Magnetic Field [uT]"]
    labels = ["Acceleration X", "Acceleration Y", "Acceleration Z", "Angular Velocity X", "Angular Velocity Y", "Angular Velocity Z", "Magnetic Field X", "Magnetic Field Y", "Magnetic Field Z"]
    data_list = [accel_data, gyro_data, mag_data]
    colors = ['blue', 'red', 'green']

    for idx, data in enumerate(data_list):
        timestamps = data[:, 0] - data[0, 0]
        timing = []
        x = data[:, 1]
        y = data[:, 2]
        z = data[:, 3]

        outpath_timing = os.path.join(plot_dirpath, f"{titles[idx].lower()}_timing.png")
        outpath_data = os.path.join(plot_dirpath, f"{titles[idx].lower()}_data.png")
        
        test_duration = (timestamps[-1] - timestamps[0]) / (1e6)
        
        for i in range(0, len(timestamps)-1):
            timing.append(timestamps[i+1] - timestamps[i])

        plt.figure()
        y_printouts, x_printouts, _ = plt.hist(timing, bins=100, edgecolor="black", color=colors[idx])
        plt.title(f"{titles[idx]} Timing Results")
        plt.xlabel(f"{titles[idx]} time between reads (us)")
        plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts)) + np.min(x_printouts), 0.9 * np.max(y_printouts), f'Maximum: {np.round(np.max(timing), 3)} us', fontsize=12)
        plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts)) + np.min(x_printouts), 0.85 * np.max(y_printouts), f'Mean: {np.round(np.mean(timing), 3)} us', fontsize=12)
        plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts)) + np.min(x_printouts), 0.8 * np.max(y_printouts), f'St. Dev.: {np.round(np.std(timing), 3)} us', fontsize=12)
        plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts)) + np.min(x_printouts), 0.75 * np.max(y_printouts), f'Total Readouts: {len(timing)}', fontsize=12)
        plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts)) + np.min(x_printouts), 0.7 * np.max(y_printouts), f'Readout Frequency: {np.round(len(timing) / test_duration, 3)} Hz', fontsize=12)
        plt.savefig(outpath_timing, format="png")
        pdf.savefig()
        ##plt.show()
        plt.close()

        plt.figure()
        plt.plot(timestamps, x, color='blue', label=labels[(3*idx)])
        plt.plot(timestamps, y, color='red', label=labels[(3*idx + 1)])
        plt.plot(timestamps, z, color='green', label=labels[(3*idx + 2)])
        plt.title(f"{titles[idx]} Readings")
        plt.xlabel("Time (us)")
        plt.ylabel(f"{ylabels[idx]}")
        plt.legend()
        plt.savefig(outpath_data, format="png")
        pdf.savefig()
        #plt.show()
        plt.close()

        print(f"{titles[idx]} plots saved")

    # Barometer has a separate structure
    bar_timestamps = bar_data[:, 0] - bar_data[0, 0]
    bar_timing = []
    bar_pressure = bar_data[:, 1]
    bar_temp = bar_data[:, 2]

    for i in range(0, len(bar_timestamps)-1):
        bar_timing.append(bar_timestamps[i+1] - bar_timestamps[i])

    bar_timing_outpath = os.path.join(plot_dirpath, "barometer_timing.png")
    bar_temp_outpath = os.path.join(plot_dirpath, "barometer_temp.png")
    bar_pres_outpath = os.path.join(plot_dirpath, "barometer_pres.png")

    test_duration = (bar_timestamps[-1] - bar_timestamps[0]) / (1e6)

    plt.figure()
    y_printouts, x_printouts, _ = plt.hist(bar_timing, bins=100, edgecolor="black", color='yellow')
    plt.title(f"Barometer Timing Results")
    plt.xlabel(f"Barometer time between reads (us)")
    plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts))+ np.min(x_printouts), 0.9 * np.max(y_printouts), f'Maximum: {np.round(np.max(bar_timing), 3)} us', fontsize=12)
    plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts)) + np.min(x_printouts), 0.85 * np.max(y_printouts), f'Mean: {np.round(np.mean(bar_timing), 3)} us', fontsize=12)
    plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts)) + np.min(x_printouts), 0.8 * np.max(y_printouts), f'St. Dev.: {np.round(np.std(bar_timing), 3)} us', fontsize=12)
    plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts)) + np.min(x_printouts), 0.75 * np.max(y_printouts), f'Total Readouts: {len(bar_timing)}', fontsize=12)
    plt.text(0.3 * (np.max(x_printouts)-np.min(x_printouts)) + np.min(x_printouts), 0.7 * np.max(y_printouts), f'Readout Frequency: {np.round(len(bar_timing) / test_duration, 3)} Hz', fontsize=12)
    plt.savefig(bar_timing_outpath, format="png")
    pdf.savefig()
    #plt.show()
    plt.close()

    plt.figure()
    plt.plot(bar_timestamps, bar_pressure, color='blue')
    plt.title(f"Barometer Pressure Readings")
    plt.xlabel("Time (us)")
    plt.ylabel(f"Pressure [Pa]")
    plt.savefig(bar_pres_outpath, format="png")
    pdf.savefig()
    #plt.show()
    plt.close()

    plt.figure()
    plt.plot(bar_timestamps, bar_temp, color='red')
    plt.title(f"Barometer Temperature Readings")
    plt.xlabel("Time (us)")
    plt.ylabel(f"Temperature [C]")
    plt.savefig(bar_temp_outpath, format="png")
    pdf.savefig()
    #plt.show()
    plt.close()

    print(f"Barometer plots saved")

################################### GPS #####################################################

def plot_gps(dirpath, plot_dirpath, pdf):
    gps_path = get_path_from_keyword(dirpath, "ZED-F9P")
    logfile = get_path_from_keyword(dirpath, "file.log")
    gps_data = read_gps_file(gps_path=gps_path, logfile=logfile)

    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.suptitle("GPS coordinates")
    axs[0].plot(gps_data["datetime"], gps_data["lon"])
    axs[0].set_ylabel("Longitude \n [deg]")
    axs[1].plot(gps_data["datetime"], gps_data["lat"])
    axs[1].set_ylabel("Latitude \n [deg]")
    axs[2].plot(gps_data["datetime"], gps_data["height"])
    axs[2].set_ylabel("Altitude \n [m]")
    axs[2].set_xlabel("Time")

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dirpath, "gps.png"))
    pdf.savefig()
    print("GPS plots saved")

    #plt.show()

################################### ADC #####################################################

HP_diode = {'Frequency [GHz]': 93,
            'Responsivity [mV/mW]': -58.55,
            'Error [mV/mW]': 0.22
        }
Eravant_diode = {'Frequency [GHz]': np.array([84,87,90,93,96,99,102]),
                'Responsivity [mV/mW]': np.array([106.29,128.73,136.59,133.19,152.43,161.78,81.64]),
                'Error [mV/mW]': np.array([2.58,4.47,5.27,5.19,8.93,10.00,8.03])
                }
Transmission_factor = {"84 GHz": -0.83,
                    "87 GHz": -0.79,
                    "90 GHz": -0.73,
                    "93 GHz": -0.68,
                    "96 GHz": -0.64,
                    "99 GHz": -0.61,
                    "102 GHz": -0.62
                    }
Coupling_factor = {"84 GHz": -20.36,
                "87 GHz": -20.26,
                "90 GHz": -20.13,
                "93 GHz": -20.10,
                "96 GHz": -20.14,
                "99 GHz": -20.25,
                "102 GHz": -20.49
                }
Attenuator_factor = {"84 GHz": -28.18,
                    "87 GHz": -29.29,
                    "90 GHz": -29.69,
                    "93 GHz": -29.98,
                    "96 GHz": -30.97,
                    "99 GHz": -31.24,
                    "102 GHz": -31.25
                    }
Gain_factor = {"84 GHz": 7.11,
            "87 GHz": 7.59,
            "90 GHz": 7.74,
            "93 GHz": 7.60,
            "96 GHz": 7.82,
            "99 GHz": 8.07,
            "102 GHz": 8.26
            }

def mW_to_dBm(pwr_mW):
    return 10*np.log10(pwr_mW)
def dBm_to_mW(pwr_dBm):
    return 10**(pwr_dBm/10)

def get_out_pwr_coupl(diode_response, responsivity, transm, coupling):
    return mW_to_dBm(diode_response/responsivity * 10**((transm-coupling)/10))

def get_output_pwr(diode_response, responsivity, transm, coupling, atten, gain):
    return get_out_pwr_coupl(diode_response, responsivity, transm, coupling) + atten + gain

def decode_bin_file(file_path):
    decoded_data = []
    try:
        with open(file_path, "rb") as f:
            lines = f.readlines()

        for line in lines:
            # Decode the line from bytes to ASCII and split
            text_line = line.decode("ascii").strip()
            if text_line:
                timestamp, value = text_line.split()
                decoded_data.append((int(timestamp), int(value)))
    except Exception as e:
        print(f"Error decoding file: {e}")

    return decoded_data                
                         
def plot_adc(dirpath, plot_dirpath, pdf):

    adc_path = get_path_from_keyword(dirpath, "ADS")
    adc_data = read_adc_file(adc_path, 16)
    # Normalize timestamps (optional but helpful for plotting)
    adc_data["timestamp"] = (adc_data["timestamp"] - adc_data["timestamp"].iloc[0]) / 1e6  # convert from ns to seconds
    adc_data["amplitude"] = adc_data["amplitude"].astype(int)

    #Compute in dBm
    output_pwr = get_output_pwr(adc_data["amplitude"].values,Eravant_diode['Responsivity [mV/mW]'][(Eravant_diode["Frequency [GHz]"] == 93)],
                            Transmission_factor['93 GHz'],
                            Coupling_factor['93 GHz'],
                            Attenuator_factor['93 GHz'],
                            Gain_factor['93 GHz'])

    fig, axs = plt.subplots(3, 1)
    fig.suptitle("ADC power")
    axs[0].plot(adc_data["timestamp"].values, adc_data["amplitude"].values)
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude \n [mV]")

    axs[1].plot(adc_data["timestamp"].values, output_pwr)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Source output power \n [dBm]")

    axs[2].hist(adc_data["timestamp"].diff(), color="orange")
    axs[2].set_xlabel("Reading time (s)")
    axs[2].set_ylabel("# of measurements")

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dirpath, "adc.png"))
    pdf.savefig()
    print("ADC plots saved")

    #plt.show()

################################################# INCLINOMETER ######################################################
color_angles = {"yaw":"cornflowerblue", "pitch":"darkorange", "roll":"forestgreen"}

def plot_inclinometer(dirpath, plot_dirpath, pdf):

    inclino_path = get_path_from_keyword(dirpath, "Kernel-100")
    logfile = get_path_from_keyword(dirpath, "file.log")

    inclino_data = read_inclino_file(inclino_path, logfile)

    timestamps = inclino_data["datetime"] if "datetime" in inclino_data.keys() else inclino_data["timestamp"]
    time_label = "Datetime" if "datetime" in inclino_data.keys() else "timestamps [s]"

    fig, axs = plt.subplots(4, 1, sharex=True)
    fig.suptitle("Inclinometer angles")
    axs[0].plot(timestamps, inclino_data["pitch"], color=color_angles["pitch"], label="Pitch")
    axs[0].set_ylabel("Pitch [deg]")
    axs[0].grid(color="gray")
    axs[0].legend()
    axs[1].plot(timestamps, inclino_data["roll"], color=color_angles["roll"], label="Roll")
    axs[1].set_ylabel("Roll [deg]")
    axs[1].grid(color="gray")
    axs[1].legend()
    axs[2].plot(timestamps, inclino_data["yaw"], color=color_angles["yaw"], label="Yaw")
    axs[2].set_ylabel("Yaw [deg]")
    axs[2].grid(color="gray")
    axs[2].legend()

    axs[3].plot(timestamps, inclino_data["Temper"], color="k", label="Temperature")
    axs[3].set_xlabel(time_label)
    axs[3].set_ylabel("Temperature [°C]")
    axs[3].grid(color="gray")
    axs[3].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(plot_dirpath, "inclinometer.png"))
    pdf.savefig()
    print("Inclinometer plots saved")
    #plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot data from payload sensors.")
    parser.add_argument("dirpath", type=str, help="Directory path to the folder containing 'sensors_data' folder and 'file.log' with sensor files.")
    parser.add_argument("-g", "--gps", action="store_true", help="Plots the GPS quick look")
    parser.add_argument("-k", "--kernel", action="store_true", help="Plots the Kernel-100 inclinometer quick look")
    parser.add_argument("-a", "--adc", action="store_true", help="Plots the ADC quick look")
    parser.add_argument("-i", "--inertial", action="store_true", help="Plots the barometer, magnetometer, accelerometer and gyroscope quick look")
    parser.add_argument("-o", "--output-dir", default="", help="Output directory to save the plots.")


    args = parser.parse_args()
    if not args.output_dir:
        args.output_dir = f"./quick_look/{os.path.basename(os.path.normpath(args.dirpath))}"

    plot_dirpath = os.path.join(args.output_dir, "plots")
    print("All plots saved in : ", plot_dirpath)
    os.makedirs(plot_dirpath, exist_ok=True)

    pdf_path = os.path.join(plot_dirpath, "summary_report.pdf")
    pdf = PdfPages(pdf_path)

    #Plot all of them if no arguments given, only the one asked if the argument are given
    if not args.gps and not args.kernel and not args.adc and not args.inertial:
         args.gps, args.kernel, args.adc, args.inertial = True, True, True, True

    if args.inertial:
        plot_inertial(args.dirpath, plot_dirpath, pdf)
    if args.gps:
        plot_gps(args.dirpath, plot_dirpath, pdf)
    if args.adc:
        plot_adc(args.dirpath, plot_dirpath, pdf)
    if args.kernel:
        plot_inclinometer(args.dirpath, plot_dirpath, pdf)

    pdf.close()
    print(f"PDF report saved to: {pdf_path}")


