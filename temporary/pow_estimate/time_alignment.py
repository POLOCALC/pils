import numpy as np
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import polars as pl

def align_adc_det(adc_data,dets_w_drone):
    time_subsec = np.zeros(len(dets_w_drone),dtype=object)
    amplitude_subsec = np.zeros(len(dets_w_drone),dtype=object)
    envelope_subsec = np.zeros(len(dets_w_drone),dtype=object)
    max_ampl_subsec = np.zeros(len(dets_w_drone),dtype=object)
    std_ampl_subsec = np.zeros(len(dets_w_drone),dtype=object)

    for i in range(len(dets_w_drone)):

        n_cross = len(dets_w_drone[i]["crossing_idx"])
        t_sub = []
        a_sub = []
        env_sub = []
        ampmax_sub = []
        std_sub = []

        for j in range(n_cross):
            time_mask_det =  (adc_data["timestamp"]>dets_w_drone[i]["t_init_cross"][j]) & (adc_data["timestamp"]<dets_w_drone[i]["t_end_cross"][j])

            t_sub.append(adc_data["timestamp"][time_mask_det])
    
            # filter the signal
            
            filtered_signal = medfilt(adc_data["amplitude"][time_mask_det], kernel_size=5)
            a_sub.append(filtered_signal)

            # Parameters: adjust based on your square wave period
            window_size = 100  # Number of samples in one square-wave high or low region

            # Initialize envelope array
            envelope = np.zeros_like(filtered_signal)

            # Process in blocks
            for start in range(0, len(filtered_signal), window_size):
                end = min(start + window_size, len(filtered_signal))
                block = filtered_signal[start:end]
                max_val = np.max(block)
                envelope[start:end] = max_val

            env_sub.append(envelope)

            ampmax_sub.append(np.mean(envelope))
            std_sub.append(np.std(envelope))

        time_subsec[i] = t_sub
        amplitude_subsec[i] = a_sub
        envelope_subsec[i] = env_sub
        max_ampl_subsec[i] = ampmax_sub
        std_ampl_subsec[i] = std_sub

    return time_subsec,amplitude_subsec,envelope_subsec,max_ampl_subsec,std_ampl_subsec


def align_gps_det(time,dist,dets_w_drone):
    time_subsec = np.zeros(len(dets_w_drone),dtype=object)
    dist_subsec = np.zeros(len(dets_w_drone),dtype=object)
    avg_time = np.zeros(len(dets_w_drone),dtype=object)
    avg_dist = np.zeros(len(dets_w_drone),dtype=object)

    for i in range(len(dets_w_drone)):

        n_cross = len(dets_w_drone[i]["crossing_idx"])
        t_sub = []
        d_sub = []
        avg_t = []
        avg_d = []

        for j in range(n_cross):
            time_mask_det =  (time>dets_w_drone[i]["t_init_cross"][j]) & (time<dets_w_drone[i]["t_end_cross"][j])

            t_sub.append(time[time_mask_det])
            d_sub.append(dist[time_mask_det])
        
            # get the average distance and the time at the average distance
            d = np.mean(dist[time_mask_det])
            t = np.mean(time[time_mask_det])
            
            avg_t.append(t)
            avg_d.append(d)

        time_subsec[i] = t_sub
        dist_subsec[i] = d_sub
        avg_time[i] = avg_t
        avg_dist[i] = avg_d

    return time_subsec,dist_subsec,avg_time,avg_dist

def identify_flight_phases(df: pl.DataFrame, window_size: int = 20) -> dict:
    t = df["timestamp"].to_numpy()
    el = df["el"].to_numpy()
    
    results = {
        "Takeoff Time": None,
        "Holding Pos Start": None,
        "Holding Pos End": None,
        "Landing Time": None
    }

    peaks, _ = find_peaks(el, height=20, prominence=1)
    
    if len(peaks) == 0:
        print("No flight peaks detected.")
        return df, results
        
    first_peak_idx = peaks[0]
    last_peak_idx = peaks[-1]
    

    pre_flight_el = el[:first_peak_idx]
    ground_data = pre_flight_el[pre_flight_el < 10]
    baseline_el = np.median(ground_data) if len(ground_data) > 0 else el[0]

    plateau_start_threshold = el[first_peak_idx] - 5.0
    is_below_plateau = pre_flight_el < plateau_start_threshold
    
    if np.any(is_below_plateau):
        # The first sample *after* it crosses the threshold
        hold_start_idx = np.where(is_below_plateau)[0][-1] + 1
        results["Holding Pos Start"] = t[hold_start_idx]


    is_ground = pre_flight_el <= (baseline_el + 1.5)
    if np.any(is_ground):
        takeoff_idx = np.where(is_ground)[0][-1]
        results["Takeoff Time"] = t[takeoff_idx]


    post_flight_el = el[last_peak_idx:]
    
    # Hold End: The moment it drops out of the plateau to start descending
    plateau_end_threshold = el[last_peak_idx] - 5.0
    is_below_plateau_end = post_flight_el < plateau_end_threshold
    
    if np.any(is_below_plateau_end):
        # The last sample *before* it crosses the threshold downward
        hold_end_idx = last_peak_idx + np.where(is_below_plateau_end)[0][0] - 1
        results["Holding Pos End"] = t[hold_end_idx]

    # Landing: The moment it touches back down to the ground baseline
    if results["Holding Pos End"] is not None:
        start_landing_search = np.where(t == results["Holding Pos End"])[0][0]
    else:
        start_landing_search = last_peak_idx
        
    post_hold_el = el[start_landing_search:]
    is_back_on_ground = post_hold_el <= (baseline_el + 1.5)
    
    if np.any(is_back_on_ground):
        landing_idx = start_landing_search + np.where(is_back_on_ground)[0][0]
        results["Landing Time"] = t[landing_idx]


    return df, results

def plot_flight_phases(df: pl.DataFrame, results: dict):
    """
    Plots Azimuth and Elevation with vertical lines for key flight phases.
    """
    # Extract data to numpy arrays for plotting
    t = df["timestamp"].to_numpy()
    azi = df["az"].to_numpy()
    elev = df["el"].to_numpy()

    # Set up the figure with two subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), dpi=120)
    fig.suptitle("Drone Flight Telemetry with Key Moments Identified", fontsize=14)

    # --- Plot 1: Azimuth ---
    ax1.plot(t, azi, label="Azimuth Data", color="#cf8c4a", linewidth=2)
    ax1.set_title("Azimuth with Flight Phases")
    ax1.set_ylabel("Azimuth (deg)")
    ax1.set_xlabel("Timestamp")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Elevation ---
    ax2.plot(t, elev, label="Elevation Data", color="#3b72a4", linewidth=2)
    ax2.set_title("Elevation with Flight Phases")
    ax2.set_ylabel("Elevation (deg)")
    ax2.set_xlabel("Timestamp")
    ax2.grid(True, alpha=0.3)

    # --- Add Vertical Lines for Key Moments ---
    # Define colors and labels for our events
    event_styles = {
        "Takeoff Time": {"color": "green", "ls": "--", "label": "Takeoff"},
        "Holding Pos Start": {"color": "purple", "ls": ":", "label": "Hold Start"},
        "Holding Pos End": {"color": "purple", "ls": ":", "label": "Hold End"},
        "Landing Time": {"color": "red", "ls": "--", "label": "Landing"}
    }

    # Loop through the results and plot lines on BOTH graphs if a timestamp was found
    for event_name, timestamp in results.items():
        if timestamp is not None:
            style = event_styles[event_name]
            # Plot on Azimuth graph
            ax1.axvline(x=timestamp, color=style["color"], linestyle=style["ls"], 
                        linewidth=1.5, label=style["label"])
            # Plot on Elevation graph
            ax2.axvline(x=timestamp, color=style["color"], linestyle=style["ls"], 
                        linewidth=1.5, label=style["label"])

    # Prevent duplicate labels in the legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="best")

    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys(), loc="best")

    plt.tight_layout()
    plt.show()

def get_scanning_range_pils(drone_data,plot=True):
    
    time_gps = drone_data["timestamp"]
    alt_gps = drone_data["altitude"]

    alt_mask = alt_gps>5400

    peaks = sp.signal.find_peaks(alt_gps.filter(alt_mask),distance=75)

    time_dist_peak = np.average(np.gradient(np.array(time_gps.filter(alt_mask))[peaks[0]]))

    scan_start_time = np.array(time_gps.filter(alt_mask))[peaks[0][0]]-time_dist_peak/2
    scan_end_time = np.array(time_gps.filter(alt_mask))[peaks[0][-1]]#+time_dist_peak/2

    if plot:
        fig = plt.figure()
        plt.plot(time_gps,alt_gps)

        plt.axvline(scan_start_time,color="green",label="scan start")
        plt.axvline(scan_end_time,color="red",label="scan end")
        plt.xlabel("Time [s]")
        plt.ylabel("Altitude [m]")
        plt.title("Drone scanning position")
        plt.legend()

        plt.show()
    
    return scan_start_time, scan_end_time

def get_scanning_range_new(DH,plot=True):
    
    time_gps = DH.payload.gps.data["timestamp"]
    alt_gps = DH.payload.gps.data["height"]
    lon_gps = DH.payload.gps.data["lon"]
    lat_gps = DH.payload.gps.data["lat"]

    alt_mask = alt_gps>5400

    peaks = sp.signal.find_peaks(alt_gps[alt_mask],distance=75)

    #for p in peaks[0]:
        #plt.axvline(np.array(time_gps[alt_mask])[p],color="red")

    time_dist_peak = np.average(np.gradient(np.array(time_gps[alt_mask])[peaks[0]]))

    scan_start_time = np.array(time_gps[alt_mask])[peaks[0][0]]-time_dist_peak/2
    scan_end_time = np.array(time_gps[alt_mask])[peaks[0][-1]]+time_dist_peak/2

    if plot:
        fig = plt.figure()
        plt.plot(time_gps,alt_gps)

        plt.axvline(scan_start_time,color="green",label="scan start")
        plt.axvline(scan_end_time,color="red",label="scan end")
        plt.xlabel("Time [s]")
        plt.ylabel("Altitude [m]")
        plt.title("Drone scanning position")
        plt.legend()

        plt.show()
    
    return scan_start_time, scan_end_time

def get_flying_range(data, litchi):
    timerange = litchi.data[litchi.data["isflying"].astype("bool")]["datetime"]
    is_flying = data["datetime"].between(timerange.iloc[0], timerange.iloc[-1])
    return data[is_flying]

def detect_peaks(data, column, negate=False, width=1, prominence=1):
    sign = -1 if negate is True else 1    
    peaks, _ = sp.signal.find_peaks(sign*data[column], width=width, prominence=prominence)
    return [data["datetime"].iloc[p] for p in peaks]

def get_scanning_range(DH, margin=15, plot=False):
    # Drone peaks
    drone_data = get_flying_range(DH.drone.data, DH.litchi)
    drone_data = drone_data.groupby("datetime").first().reset_index()

    alt_peaks = detect_peaks(drone_data, "RTKdata:Hmsl_P",negate=True,  width=5, prominence=1e-7)
    lat_peaks = detect_peaks(drone_data, "RTKdata:Lat_P", negate=True, width=10, prominence=1e-4)
    lon_peaks = detect_peaks(drone_data, "RTKdata:Lon_P", negate=True, width=10, prominence=1e-4)

    # Inclino peaks
    inclino_data = get_flying_range(DH.payload.inclino.data, DH.litchi)
    inclino_peaks = detect_peaks(inclino_data, "pitch", negate=True, width=50, prominence=3)

    # Barometer peaks
    baro_data = get_flying_range(DH.payload.baro.data, DH.litchi)
    baro_peaks = detect_peaks(baro_data, "pressure", negate=False, width=1, prominence=2)

    all_peaks = alt_peaks + lat_peaks + lon_peaks + inclino_peaks + baro_peaks

    # Convert to pandas Series of datetime64[ns]
    ts_series = pd.Series(all_peaks).sort_values().reset_index(drop=True)

    # Convert to seconds since first timestamp
    seconds = (ts_series - ts_series.min()).dt.total_seconds().values

    # Find timestamps that have at least another timestamp within 1 second
    keep_mask = np.zeros(len(seconds), dtype=bool)

    for i, t in enumerate(seconds):
        # Check if any other timestamp is within 1 second
        twin = 5
        if np.any(np.abs(seconds - t) <= twin) and np.sum(np.abs(seconds - t) <= twin) > 1:
            keep_mask[i] = True

    # Filter consensus peaks
    consensus_peaks = ts_series[keep_mask]

    # Compute scanning range
    if not consensus_peaks.empty:
        scanning_range = (consensus_peaks.min(), consensus_peaks.max())
    else:
        scanning_range = None

    scan_start, scan_end = scanning_range[0] - pd.Timedelta(seconds=margin), scanning_range[1] + pd.Timedelta(seconds=margin)

    if plot:
        baro_data = get_flying_range(DH.payload.baro.data, DH.litchi)
        plt.figure()
        plt.plot(baro_data["datetime"], baro_data["pressure"])

        # Add vertical lines for each type of peak
        for dt in all_peaks:
            plt.axvline(dt, color="k", ls="--", alpha=0.2)
        plt.axvline(scan_start, color="r", ls="-")
        plt.axvline(scan_end, color="r", ls="-")
        plt.show()

    return scan_start, scan_end , all_peaks
















    