import re
import yaml
import sys
from pathlib import Path
import copy
 
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/home/jupyter-ludovico/stout")

from stout.services.inventory.service import InventoryService
from stout.services.measurements.service import MeasurementService
import datetime

import argparse


def mW_to_dBm(pwr_mW):
    return 10*np.log10(pwr_mW)

def dBm_to_mW(pwr_dBm):
    return 10**(pwr_dBm/10)
    

def read_raw_data(raw_data_path):
    """
    Read the raw data and create a dictionary with the following structure:
    - Frequency [GHz]
        - Temperature [°C]
            - Attenuator control Voltage [V]
            - Power [mW]
            - Diode output Voltage [mV]
    """
    processed_dict = {}
    #print(raw_data_path)

    with h5py.File(raw_data_path, "r") as f:
        for freq_grp_name, freq_group in f.items():

            processed_dict.update({freq_grp_name: {}})

            for temp_grp_name, temp_group in freq_group.items():

                processed_dict[freq_grp_name].update({temp_grp_name: {}})

                cv_vec = []
                avg_pwr_vec = []
                err_pwr_vec = []
                avg_outV_vec = []
                err_outV_vec = []
                pwr_baseline_vec = []
                pm_range_vec = []

                for cv_grp_name, cv_group in temp_group.items():
                    #print(cv_grp_name)
                    #print(dict(cv_group.attrs))

                    cv_vec.append(cv_group.attrs["control_voltage"])
                    pm_range_vec.append(cv_group.attrs["PM range"])

                    time = cv_group["Time [s]"][()]
                    power = cv_group["Power [mW]"][()]
                    diode_out = cv_group["Diode output [mV]"][()]

                    # analysis for low power measurements
                    if(len(time)>100):
                        # arrays for on and off region:
                        pwr_on = []
                        pwr_off = []
                        diode_on = []
                        diode_off = []
                        diff_pwr = []
                        diff_outV = []
                        zero_pwr = []
                    
                        # get the number of on/off for low power measurements
                        n_rep = int(len(time))/30

                        #print(f"{n_rep} repeated meas for {cv_group.attrs["control_voltage"]}V")

                        n=0
                        while(n<n_rep):
                            pwr_off.append(power[n*30:(n+1)*30] )
                            pwr_on.append(power[(n+1)*30:(n+2)*30] )
            
                            diode_off.append(diode_out[n*30:(n+1)*30])
                            diode_on.append(diode_out[(n+1)*30:(n+2)*30])
            
                            n = n+2

                        for i in range(len(pwr_on)):
                            diff_pwr.append(np.mean(np.asarray(pwr_on[i])[pwr_on[i]!=-np.inf]) - np.mean(np.asarray(pwr_off[i])[pwr_off[i]!=-np.inf]))
                            diff_outV.append(np.mean(np.asarray(diode_on[i])[diode_on[i]!=-np.inf]) - np.mean(np.asarray(diode_off[i])[diode_off[i]!=-np.inf]))

                            zero_pwr.append(np.mean(np.asarray(pwr_off[i])[pwr_off[i]!=-np.inf]))

                        diff_pwr = np.asarray(diff_pwr)
                        diff_outV = np.asarray(diff_outV)
                        zero_pwr = np.asarray(zero_pwr)
                    
                        avg_pwr_vec.append(np.mean(diff_pwr[diff_pwr>0]))
                        err_pwr_vec.append(np.std(diff_pwr[diff_pwr>0]))
                        avg_outV_vec.append(np.mean(diff_outV[diff_outV>0]))
                        err_outV_vec.append(np.std(diff_outV[diff_outV>0]))
                        pwr_baseline_vec.append(np.mean(zero_pwr))

                    else:
                        avg_pwr_vec.append(np.mean(power[power!=-np.inf]))
                        err_pwr_vec.append(np.std(power[power!=-np.inf]))
                        avg_outV_vec.append(np.mean(diode_out))
                        err_outV_vec.append(np.std(diode_out))
                        pwr_baseline_vec.append(0)

                processed_dict[freq_grp_name][temp_grp_name].update({
                    "Control Voltage [V]": np.asarray(cv_vec),
                    "Power [mW]": np.asarray(avg_pwr_vec),
                    "err power [mW]": np.asarray(err_pwr_vec),
                    "Diode output [mV]": np.asarray(avg_outV_vec),
                    "err diode output [mV]": np.asarray(err_outV_vec),
                    "power baseline [mW]": np.asarray(pwr_baseline_vec),    
                    "PM range [mW]": np.asarray(pm_range_vec)
                })

    return processed_dict
    

def load_notes(notes_path):
    with open(notes_path, "r", encoding="utf-8") as f:
        content = f.read()

    # extract the yaml bloc within: "---"
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        raise ValueError("No configuration block found in the notes file.")

    front_matter = yaml.safe_load(match.group(1))
    
    return front_matter["config"]


def get_aux_item_charact(aux_items):
    """
    Get a list of the auxiliary items used in the measure and return their characterization.
    """

    db_dirpath = Path("/mnt/data/POLOCALC/lab")
    items_charact = []

    wg_transmission = {}
    
    for ai in aux_items:
        
        if ai["category"]=="coupler":
            
            ai_band = ai["specifications"]["output_band"]
            item_path = db_dirpath / ai_band / ai["category"] / ai["name"]
            
            transmission_path = item_path / "transmission"
            directivity_path = item_path / "directivity"
            coupling_path = item_path / "coupling_performance"

            try:
                coupl_transmission = pd.read_csv(sorted(p for p in transmission_path.iterdir() if p.is_file() and p.name.lower().endswith(".csv"))[0])
                coupl_directivity = pd.read_csv(sorted(p for p in directivity_path.iterdir() if p.is_file() and p.name.lower().endswith(".csv"))[0])
                coupl_coupling = pd.read_csv(sorted(p for p in coupling_path.iterdir() if p.is_file() and p.name.lower().endswith(".csv"))[0])        

                coupler_charact = {"Transmission": coupl_transmission,
                                   "Coupling": coupl_coupling,
                                   "Directivity": coupl_directivity}

            except:
                raise ValueError("Coupler path not found")

        elif((ai["category"]=="straight") | (ai["category"]=="stainless_steel_wg")):
            ai_band = ai["specifications"]["input_band"]
            transmission_path = db_dirpath / ai_band / ai["category"] / ai["name"] / "transmission"

            try:
                wg_transm = pd.read_csv(sorted(p for p in transmission_path.iterdir() if p.is_file() and p.name.lower().endswith(".csv"))[0])
                wg_transmission.update({ai["name"]: wg_transm})
            except:
                raise ValueError("Waveguide transmission path not found")

    return coupler_charact,wg_transmission
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load multiplier output power and diode responsity raw data and process them.")
    parser.add_argument("-d","--dirpath",type=str,help="Path to the raw data directory. Could be both the multiplier or the detector path.")
    args = parser.parse_args()

    path = Path(args.dirpath)

    #print(path.iterdir())

    raw_files = sorted(p for p in path.iterdir() if p.is_file() and p.name.lower().endswith("_raw.h5py"))
    notes_file = sorted(p for p in path.iterdir() if p.is_file() and p.name.lower().endswith(".md"))

    #print(raw_files[0].name)
    #print(notes_file)

    # load the notes file
    config = load_notes(path/notes_file[0])

    multiplier_id = config["components"]["multiplier"]["item_id"]
    diode_id = config["components"]["detector"]["item_id"]

    coupler_id = config["auxiliary_components"]["directional_coupler"]["item_id"]
    waveguides = config["auxiliary_components"]["waveguides"]

    # get the items from stout:
    inv_service = InventoryService()

    aux_items = []
    for k in waveguides.keys():
        wg_id = waveguides[k]["item_id"]
        aux_items.append(inv_service.get_item_by_id(item_id=wg_id))

    coupler = inv_service.get_item_by_id(item_id=coupler_id)
    aux_items.extend([coupler])

    multiplier_item = inv_service.get_item_by_id(item_id=multiplier_id)
    diode_item = inv_service.get_item_by_id(item_id=diode_id)

    print("\nProcessing data for the following items:\n\n"multiplier_item,"\n\n",diode_item)

    #print(aux_items)

    coupler_charact, wg_transmission = get_aux_item_charact(aux_items)

    #print(wg_transmission)

    # load the data:
    if len(raw_files)==1:
        processed_dict = read_raw_data(path/raw_files[0])
    elif len(raw_files)==0:
        raise ValueError("No raw data found.")
        exit()
    else:
        raise ValueError("Multiple raw measurements have been found. Still not implemented the data processing for this case.")
        """
        processed_data_array = []
        for rf in raw_files: # the lenght of the raw_files indicates the number of measurements performed
            processed_data_array.append(path/read_raw_data(rf))
        """

    #print(processed_dict)

    corrected_dict = {}

    # correct the power by the waveguide attenuation factor
    for freqs in processed_dict.keys():
        int_freq = int(freqs.split(" GHz")[0])
        #print(int_freq)

        corrected_dict.update({freqs: {}})
        
        for temps in processed_dict[freqs].keys():
            #print(temps)
            # correct the power for the zero power baseline
            processed_dict[freqs][temps]["Power [mW]"][np.asarray(processed_dict[freqs][temps]["power baseline [mW]"])==0] -= np.mean(np.asarray(processed_dict[freqs][temps]["power baseline [mW]"])[np.asarray(processed_dict[freqs][temps]["power baseline [mW]"])!=0])

            # correct the power for the waveguide attenuation
            coupl_transm = np.asarray(coupler_charact["Transmission"]["S12 avg [dB]"])[np.asarray(coupler_charact["Transmission"]["Frequency"]==int_freq)][0]
            err_coupl_transm = np.asarray(coupler_charact["Transmission"]["S12 std [dB]"])[np.asarray(coupler_charact["Transmission"]["Frequency"]==int_freq)][0]

            coupl_factor = np.asarray(coupler_charact["Coupling"]["S12 avg [dB]"])[np.asarray(coupler_charact["Coupling"]["Frequency"]==int_freq)][0]
            err_coupl_factor = np.asarray(coupler_charact["Coupling"]["S12 std [dB]"])[np.asarray(coupler_charact["Coupling"]["Frequency"]==int_freq)][0]

            wg_transm_fact = 0
            err_wg_transm = 0
            for wg_k in wg_transmission.keys():
                wg_t = np.asarray(wg_transmission[wg_k]["S12 avg [dB]"])[np.asarray(wg_transmission[wg_k]["Frequency"]==int_freq)][0]
                err_wg_t = np.asarray(wg_transmission[wg_k]["S12 std [dB]"])[np.asarray(wg_transmission[wg_k]["Frequency"]==int_freq)][0]
                wg_transm_fact += wg_t
                err_wg_transm =+ err_wg_t**2

            err_wg_transm = np.sqrt(err_wg_transm)

            # conversion to dB 
            meas_pwr_dBm = mW_to_dBm(processed_dict[freqs][temps]["Power [mW]"])
            meas_pwr_dBm -= (wg_transm_fact - 0.2) #0.2 factor from PM5B specs; output power from the coupler
            
            # uncertainty on wg_transm is negligible -> propagating only for the multiplication factor

            #processed_dict[freqs][temps]["err power [mW]"] *= 10**(abs(wg_transm_fact - 0.2)/10)
            corrected_pwr = dBm_to_mW(meas_pwr_dBm) # coupler output power
            corrected_err = processed_dict[freqs][temps]["err power [mW]"] * 10**(abs(wg_transm_fact - 0.2)/10)

            #print(coupl_factor,coupl_transm)

            # compute the diode input power:
            diode_input_pwr = corrected_pwr * 10**((coupl_factor-coupl_transm)/10)
            diode_input_err = diode_input_pwr * np.sqrt((corrected_err/corrected_pwr)**2 + (np.log(10)/10)**2 * (err_coupl_factor**2 + err_coupl_transm**2))

            #print(diode_input_pwr,"\n",diode_input_err)

            corrected_dict[freqs].update(
                {temps: {
                    "Control Voltage [V]": processed_dict[freqs][temps]["Control Voltage [V]"],
                    "Power [mW]": corrected_pwr,
                    "err power [mW]": corrected_err,
                    "Power [dBm]": meas_pwr_dBm,
                    "Diode output [mV]": processed_dict[freqs][temps]["Diode output [mV]"],
                    "err diode output [mV]": processed_dict[freqs][temps]["err diode output [mV]"],
                    "Diode input pwr [mW]": diode_input_pwr,
                    "err diode input [mW]": diode_input_err,
                    }
                }
            )

    #print(corrected_dict.items())

    # save output file both for the diode detector and for the frequency multiplier:
    output_paths = [f"/mnt/data/POLOCALC/lab/{multiplier_item["specifications"]["output_band"]}/multiplier/{multiplier_item["name"]}/output_power/{raw_files[0].name.split("_raw")[0]}_processed.h5py",
                    f"/mnt/data/POLOCALC/lab/{diode_item["specifications"]["input_band"]}/detector/{diode_item["name"]}/responsivity/{raw_files[0].name.split("_raw")[0]}_processed.h5py"]

    for op in output_paths:
        
        with h5py.File(op, "w") as of:

            for freqs in corrected_dict.keys():
        
                int_freq = int(freqs.split(" GHz")[0])
                grp_freq = of.create_group(freqs)

                if freqs is not None:
                    grp_freq.attrs["Frequency"] = int_freq
                    grp_freq.attrs["units"] = "GHz"
        
                for temps in corrected_dict[freqs].keys():
                    int_temp = int(temps.split(" C")[0])
                    grp_temp = grp_freq.create_group(temps)

                    if temps is not None:
                        grp_temp.attrs["Temperature"] = int_temp
                        grp_temp.attrs["units"] = "°C"

                    for d_keys in corrected_dict[freqs][temps].keys():
                        grp_temp.create_dataset(d_keys, data=corrected_dict[freqs][temps][d_keys])
                    


    
                

    

    


