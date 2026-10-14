import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy
import argparse
from scipy.signal import medfilt
from scipy.interpolate import RegularGridInterpolator
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timezone
import polars as pl
import numpy as np
from scipy.signal import find_peaks
from glob import glob
import os
from tqdm import tqdm
from pathlib import Path
import h5py

from pils.loader.path import PathLoader
from pils.loader.stout import StoutLoader
from pils.flight import Flight
from pils.analyze.azel import AZELAnalysis
from pils.loader.stout import StoutLoader

import pointing_correction_tools
import interp_error_prop
import time_alignment

import sys
sys.path.insert(0, "/home/jupyter-ludovico/stout")
from stout.services.campaigns.service import CampaignService

def filter_to_common(data_list, cv_list, cv_common):
        out = []
        for data_row, cv in zip(data_list, cv_list):
            mask = np.isin(cv, cv_common)
            out.append(data_row[mask])
        return np.array(out)

def h5_to_dict(h5obj):
    """Recursively convert an h5py Group/Dataset into a nested dict."""
    result = {}
    for key, item in h5obj.items():
        if isinstance(item, h5py.Group):
            result[key] = h5_to_dict(item)          # recurse into subgroup
        else:  # h5py.Dataset
            result[key] = item[:]                    # load array into memory
    return result

def load_data(file_path):
    with h5py.File(file_path, "r") as f:
        return h5_to_dict(f)

def get_beam_solid_angle(fwhm):
    """
    fwhm: should be in radians
    """
    return 2 *  np.pi * (fwhm*u.rad  / np.sqrt(8*np.log(2)))**2

def get_detector_gain(beam_solid_angle):
    """
    Return the gain of a single detector in dBi
    (assume u.dB is dBi here since we multiply with 4pi)
    """
    fw_gain = 10*np.log10(4*np.pi / beam_solid_angle.value)
    return (fw_gain) * u.dB

def get_detector_power(output_power, err_output_pwr, detector_gain, freq, optical_efficiency, distance=500, duty_cycle=2/np.pi):
    """
    Include the distance and the optical efficiency
    and return receiver power in dBm
    freq: assumed in GHz
    """
    fspl = (const.c.value / (4 * np.pi * freq * u.GHz.to(u.Hz) * (distance*u.meter).value))**2
    fspl_dBm = 10*np.log10(fspl)
    fspl_dBm *= u.dB       
    optical_efficiency_loss = (10*np.log10(optical_efficiency * duty_cycle))*u.dB

    expctd_pwr_det = detector_gain + output_power + fspl_dBm + optical_efficiency_loss
    err_expctd_pwr_det = err_output_pwr * 10**((fspl_dBm.value + detector_gain.value + optical_efficiency_loss.value)/10)
    
    return expctd_pwr_det, err_expctd_pwr_det

def get_drone_power(effective_diameter, beam_solid_angle, optical_efficiency, temp=300, distance=500, bandpass=30):
    """
    temp: assumed commonly a perfect blackbody of 300 K 
    effective_diameter: for POLOCALC/hovercal is 0.5 m
    bandpass: assumed in GHz
    """
    
    effective_angular_diameter = effective_diameter*u.meter / distance*u.meter
    dilution_factor = np.pi * (effective_angular_diameter.value / 2) ** 2 / beam_solid_angle.value

    return (10*np.log10(const.k_B.to(u.mJ/u.K).value * (temp*u.K).value * (bandpass*u.GHz.to(u.Hz) * optical_efficiency * dilution_factor)))*u.dB

def get_drone_T_RJ(effective_diameter, beam_solid_angle, optical_efficiency, temp=300, distance=500, bandpass=30):
    """
    temp: assumed commonly a perfect blackbody of 300 K 
    effective_diameter: for POLOCALC/hovercal is 0.5 m
    bandpass: assumed in GHz
    """
    
    effective_angular_diameter = effective_diameter*u.meter / distance*u.meter
    dilution_factor = np.pi * (effective_angular_diameter.value / 2) ** 2 / beam_solid_angle.value

    return temp*dilution_factor

def mW_to_dBm(pwr_mW):
    return 10*np.log10(pwr_mW)

def dBm_to_mW(pwr_dBm):
    return 10**(pwr_dBm/10)

def get_item_from_payload(payload,item_category):
    """
    Returns the desired item given the flying payload object and the item category string.
    """
    items_list = []

    try:
    
        for i in payload["items"]:
                if i["category"]==item_category:
                    items_list.append(i)

        if len(items_list)==1:
            return items_list[0]

        elif len(items_list)>1:
            return items_list

        elif len(items_list)==0:
            raise ValueError("No items with given item_category found")

    except Exception as e:
        raise ValueError(f"Error in finding {item_category} in the paylaod: {e}")

def get_item_lab_charact_datapath(dB_dirpath,item,meas_type,key):
    """
    Returns the path to the lab characterization database of the item.
    Input:
        dB_dirpath (str): general path to the database (usually 'mnt/data/POLOCALC/lab')
        item (dB object): database item object present in the paylaod
        meas_type (str): item measurement type
        key (str): key to be find in the files to get the correct characterization file
    """
    try:
        item_spec = item["specifications"]
        band_mask = ["band" in p for p in list(item_spec.keys())]
        band_key = np.asarray(list(item_spec.keys()))[band_mask][0]
        item_charact_path = dB_dirpath / item_spec[band_key] / item["category"] / item["name"] / meas_type

        item_charact_file = sorted(p for p in item_charact_path.iterdir() if p.is_file() and key in p.name.lower())[0]

    except Exception as e:
        raise ValueError(f"Error in loading the lab characterization file for {item['name']}: \n{e}")
        
    return item_charact_file
        
def get_item_lab_charact(dB_dirpath,item,meas_type,key):
    """
    Returns the path to the lab characterization database of the item.
    Input:
        dB_dirpath (str): general path to the database (usually 'mnt/data/POLOCALC/lab')
        item (dB object): database item object present in the paylaod
        meas_type (str): item measurement type
        key (str): key to be find in the files to get the correct characterization file
    """
    try:
        item_spec = item["specifications"]
        band_mask = ["band" in p for p in list(item_spec.keys())]
        band_key = np.asarray(list(item_spec.keys()))[band_mask][0]
        item_charact_path = dB_dirpath / item_spec[band_key] / item["category"] / item["name"] / meas_type

        item_charact_file = sorted(p for p in item_charact_path.iterdir() if p.is_file() and key in p.name.lower())[0]

        #print(item_charact_file)

        if item_charact_file.name.lower().endswith(".csv"):
            charact_file = pd.read_csv(item_charact_file)

        elif item_charact_file.name.lower().endswith((".h5py", ".hdf5")):
            charact_file = load_data(item_charact_file)

        else: 
            raise ValueError(f"{item_charact_file} format not supported.")

    except Exception as e:
        raise ValueError(f"Error in loading the lab characterization file for {item['name']}: \n{e}")
        
    return charact_file

def get_out_pwr_coupl_Interp1d(in_pwr_diode,std_in_pwr_diode,coupler):

    # estimate the output power from the directional coupler taking into account the transmissivity and the directivity of the coupler
    out_pwr_coupl_dBm = mW_to_dBm(in_pwr_diode * 10**((coupler["Transmission [dB]"]-coupler["Coupling [dB]"])/10))
    # assuming negligible the uncertainties on the coupling and transmission factor of the coupler
    std_out_pwr_coupl_mW = std_in_pwr_diode * 10**((coupler["Transmission [dB]"]-coupler["Coupling [dB]"])/10) 
    
    return out_pwr_coupl_dBm,std_out_pwr_coupl_mW
    

def get_output_pwr_Interp1d(in_pwr_diode,std_in_pwr_diode,RF_components):

    out_pwr_coupl_dBm,std_out_pwr_coupl_mW = get_out_pwr_coupl_Interp1d(in_pwr_diode,std_in_pwr_diode,RF_components["coupler"])

    attenuation = 0
    for att in RF_components["attenuators"]:
        attenuation += att["Attenuation [dB]"]

    antenna_gain = RF_components["antenna"]["Gain [dBi]"]
    antenna_IL = RF_components["antenna"]["Insertion Loss [dB]"]
    antenna_ML = mW_to_dBm(1-dBm_to_mW(RF_components["antenna"]["Impedance Match [dB]"]))
    
    source_out_pwr_dBm = out_pwr_coupl_dBm + attenuation + antenna_gain + antenna_IL + antenna_ML
    source_out_pwr_std_mW = std_out_pwr_coupl_mW * 10**((attenuation + antenna_gain + antenna_IL + antenna_ML)/10)
    
    return source_out_pwr_dBm,source_out_pwr_std_mW

# ------------------------------------------------------------------------------------------------------------------------------------------------------


def compute_power_on_det(flight,campaign,payload,tele,ws,solid_angle="gab",bandpass_str="gab",out_dir=None,plot=True):
    """
    Computes the expected power on the detectors of the telescope.
    Input parameters:
    
        - flight (flight object): object that contains all the data of the flight.
        - campaign (str): campaign id.
        - payload(obj): Payload object (dict) containing the flying payload information.
        - tele (str): targeted telescope.
        - solid_angle (str): beam solid angle for the telescope detectors. Can be 'SO' or 'gab'. Default: 'gab'
        - bandpass_str (str): bandpass of the telescope. Needs to be consistent to 'solid_angle'. Can be 'SO' or 'gab'. Default: 'gab'
        - plot(bool): plot the data analysis ongoing
    """

    if out_dir is None:
        savefig = False
    else:
        out_dir_fig = f"{out_dir}/figures"
        if not os.path.exists(out_dir_fig):
            os.makedirs(out_dir_fig)
        savefig = True
    
    # set-up sats detectors: --------------------------------------------------------------------------------------------------------------
    fwhms = {'WR10':None, 'WR06':None} #in arcmins
    freqs = {'WR10':None, 'WR06':None}
    #pre_c_omegas = {'f090':136*1e-06, 'f150':51*1e-06}

    source_freq = flight.flight_info["flight_info"]["frequency"]
    flight_ID = flight.flight_info["flight_name"].split("flight_")[-1]

    multiplier = get_item_from_payload(payload,item_category="multiplier")
    band = multiplier["specifications"]["output_band"]
    freqs[band] = source_freq

    print("\n",freqs,"\n")

    if "satp" in tele:
        print("\nTelescope name:",tele)
        # Get beam solid angle
        if band=="WR10":
            
            if (tele=="satp3") & (ws=="ws5"):
                fwhms[band] = 28.6 #arcmin, from SO MF SAT commissioning paper
                print("\nUsing the avg beam from MF SAT commissioning paper:",fwhms[band]," arcmin\n")
                omega_rad = np.deg2rad(fwhms[band]/60) 
                omega = get_beam_solid_angle(omega_rad)

            else:
                
                fwhm_df = pd.read_csv(f"/home/jupyter-ludovico/optical efficiency/beams/{tele}/fwhms_{tele}_{ws}.csv")
        
                if((source_freq!=90) & (source_freq!=96)):
                    fwhms[band] = fwhm_df[fwhm_df["freq"]==source_freq]["fwhm"].values[0]

                    # compute the beam solid angle from drone characterization
                    if tele=="satp1":
                        beam_path = f"/home/jupyter-ludovico/optical efficiency/beams/{tele}/coadded_invw_fitted_full_unsmoothed_prof_T_{tele}_{int(source_freq)}_{ws}_mid.txt"
                        angle,power = np.loadtxt(beam_path,unpack=True)
                        angle_deg = angle/60
                        log_pow = 10*np.log10(power/max(power))
                    elif tele =="satp3":
                        beam_path = f"/home/jupyter-ludovico/optical efficiency/beams/{tele}/coadded_unsmoothed_prof_T_{tele}_{int(source_freq)}_{ws}_mid.txt"
                        angle,power = np.loadtxt(beam_path,unpack=True)
                        angle_deg = np.rad2deg(angle)
                        log_pow = 10*np.log10(power/max(power))

                    else:
                        raise ValueError("Invalid telescope name")

                    norm_pow = power/max(power)
                    theta = np.deg2rad(angle_deg)

                    # integrate over the angular distance
                    integrand = norm_pow * np.sin(theta)
                    integral_trapz = scipy.integrate.trapezoid(integrand,theta)
                    omega = 2*np.pi*integral_trapz *u.rad**2

                else:
                    print("\nusing average band fwhm\n")
                    fwhms[band] = np.mean(np.asarray(fwhm_df["fwhm"].values))
                
                    omega_rad = np.deg2rad(fwhms[band]/60) 
                    omega = get_beam_solid_angle(omega_rad)

        elif band=="WR06":

            if tele=="satp1":
                fwhms[band] = 12.1 # arcmin
                omega_rad = np.deg2rad(fwhms[band]/60) 
                omega = get_beam_solid_angle(omega_rad)

            elif tele=="satp3":
                fwhms[band] = 12.7 # arcmin
                omega_rad = np.deg2rad(fwhms[band]/60) 
                omega = get_beam_solid_angle(omega_rad)

            else:
                raise ValueError("Invalid telescope name")

        else:
            raise ValueError("Invalid frequency band or frequency band not yet implemented")
        
    else:
        raise ValueError("Wrong telescope name")

    print("\nfwhm:",fwhms,"\n")

    detector_gain = get_detector_gain(omega)
    print("\nDetector gain",detector_gain,"\n")

    # synchronize data:
    flight.sync(use_rtk_data=True, common_time=True,
                target_rate={"drone": 100,
                            "litchi": 100,
                            "inclinometer":100,
                            "gps":100,
                            })
    sync_data= flight.sync_data

    drone_nan_mask = np.isfinite(np.array(sync_data["drone"]["latitude"]))
    
    for k in sync_data.keys():
        if(k!="reference_gps"):
            sync_data[k] = sync_data[k].filter(drone_nan_mask)

    # check if there is any unvalid position:
    mask_valid_pos = (sync_data["drone"]["latitude"] != 0) & sync_data["drone"]["longitude"].is_not_null()
    
    for k in sync_data.keys():
        if(k!="reference_gps"):
            sync_data[k] = sync_data[k].filter(mask_valid_pos)
    
    common_timestamps = np.array(sync_data["drone"]["timestamp"])

    # set-up the scanning properties: --------------------------------------------------------------------------------------------------------------------------------------
    # generate azel and get the init and end scanning times:
    try:
        azel = AZELAnalysis(flight)
        azel_result = azel.run_analysis(tele, dji_broadcast_geod={'lat': -22.9597732, 'lon': -67.7866847, 'alt': 5173.020})
        azel_data = azel_result.azel_data
        
    except Exception as e:
        raise ValueError(f"Error in creating the azel file for flight {f_ID}: {e}")
    
    # cut the data only for the scan duration
    scan_start_time, scan_end_time = time_alignment.get_scanning_range_pils(sync_data["drone"],plot=False)
    scan_mask = (common_timestamps > scan_start_time) & (common_timestamps < scan_end_time)
    
    for k in sync_data.keys():
        if(k!="reference_gps"):
            sync_data[k] = sync_data[k].filter(scan_mask)

    azel_data = azel_data.filter(scan_mask)

    dist = np.sqrt(azel_data["E"]**2 + azel_data["N"]**2 + azel_data["U"]**2)

    # get the pointing offset power corrections: -----------------------------------------------------------------------

    # get the pointing :
    gimbal_yaw = sync_data["litchi"]["yaw(deg)"]+sync_data["litchi"]["gimbalYawRaw"]/10
    gimbal_pitch = sync_data["litchi"]["gimbalPitchRaw"]/10
    gimbal_roll = sync_data["litchi"]["gimbalRollRaw"]/10

    poi = pointing_correction_tools.compute_POI(np.asarray(azel_data["E"]),np.asarray(azel_data["N"]),np.asarray(azel_data["U"]), 
                                                np.asarray(gimbal_yaw),
                                                np.asarray(gimbal_pitch),
                                                np.asarray(sync_data["inclinometer"]["roll"]))

    tele_newframe = np.zeros([len(poi),3])
    poi_newframe = np.zeros([len(poi),3])
    rel_yaw,rel_pitch,rel_roll = np.zeros([3,len(poi)])

    # build the ENU vector for the drone
    ENU_drone = np.array([azel_data["E"],azel_data["N"],azel_data["U"]])

    # roll_array
    inclino_roll = sync_data["inclinometer"]["roll"]

    for i in range(len(poi)):
        # build the new rf centered on the drone and looking at the poi at the time t:
        T = pointing_correction_tools.build_frame(ENU_drone.T[i],poi[i],np.array(inclino_roll)[i])
        
        # move the telescope and the poi in the new rf
        tele_newframe[i] = pointing_correction_tools.world_to_local(T,np.zeros(3))
        poi_newframe[i] = pointing_correction_tools.world_to_local(T,poi[i])
        
        # compute the relative angles: yaw->az ; pitch->el
        rel_yaw[i],rel_pitch[i],rel_roll[i] = pointing_correction_tools.relative_yaw_pitch_roll(tele_newframe[i],poi_newframe[i])

    # model the 2D Gaussian beam of the source:
    antenna = get_item_from_payload(payload,item_category="antenna")

    dB_dirpath = Path("/mnt/data/POLOCALC/lab")
    
    fwhm_antenna = get_item_lab_charact(dB_dirpath=dB_dirpath,item=antenna,meas_type="radiation_pattern",key="fwhm_processed")
        
    # --- Coordinate grid ---
    x = np.linspace(-90, 90, 501)
    y = np.linspace(-90, 90, 501)
    
    # --- 2D Gaussian beam ---

    # get the correct frequency:
    fwhm_H_freq_mask = fwhm_antenna["Hplane"]["Frequency [GHz]"]==freqs[band]
    
    fwhm_H = fwhm_antenna["Hplane"]["FWHM [deg]"][fwhm_H_freq_mask]
    err_fwhm_H = fwhm_antenna["Hplane"]["err FWHM [deg]"][fwhm_H_freq_mask]

    fwhm_E_freq_mask = fwhm_antenna["Eplane"]["Frequency [GHz]"]==freqs[band]
    
    fwhm_E = fwhm_antenna["Eplane"]["FWHM [deg]"][fwhm_E_freq_mask]
    err_fwhm_E = fwhm_antenna["Eplane"]["err FWHM [deg]"][fwhm_E_freq_mask]
    
    beam_dB = 10*np.log10(pointing_correction_tools.beam_2D_gauss(x,y,fwhm_H,fwhm_E))
    beam_dB = np.maximum(beam_dB, -60)

    # interpolating the 2D linear beam to estimate values
    beam_lin_interp = scipy.interpolate.RegularGridInterpolator((y,x),pointing_correction_tools.beam_2D_gauss(x,y,fwhm_H,fwhm_E),
                                                                method='linear',bounds_error=False,fill_value=0.0)
    # compute power corrections:
    gain_losses = np.zeros(len(poi))

    for i in range(len(poi)):
        gain_losses[i] = 10*np.log10(beam_lin_interp([-np.rad2deg(rel_yaw[i]),-np.rad2deg(rel_pitch[i])])[0]) 

    if plot:
        # distance plot -------------------------------------------------------------------
        fig = plt.figure()
        plt.plot(common_timestamps[scan_mask],dist,color="orange")
        plt.xlabel("Time [s]",size=14)
        plt.title(f"Distance drone-{tele}",size=15)
        plt.ylabel("Distance [m]",size=14)
        plt.tick_params(axis="both",labelsize=14)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(f"{out_dir_fig}/Distance_drone-{tele}.png",dpi=300)
            
        plt.show()

        # pointing corrections plot -----------------------------------------------------------
        fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(22,7))

        ax[0].scatter(poi[:,0], poi[:,1], s=.1,)
        ax[0].scatter(0,0,label=tele,color="red") 
        ax[0].set_xlabel("E [m]",size=14)
        ax[0].set_ylabel("N [m]",size=14)
        ax[0].set_title("Source pointing",size=15)
        ax[0].grid()
        ax[0].legend(fontsize=14)
        ax[0].tick_params(axis="both",labelsize=12)
        
        im = ax[1].imshow(beam_dB, extent=[x.min(), x.max(), y.min(), y.max()],
                           origin='lower', cmap='inferno',vmin=-10)

        cbar = fig.colorbar(im,ax=ax[0])
        cbar.set_label("Power [dB]",fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        ax[1].set_xlabel('H-plane cuts [deg]',size=14)
        ax[1].set_ylabel('E-plane cuts [deg]',size=14)
        ax[1].set_title('Beam portion hitting the telescope',size=15)
        ax[1].scatter(-np.rad2deg(rel_yaw),-np.rad2deg(rel_pitch), c='red', s=50, edgecolors='black')
        ax[1].set_xlim(-20,20)
        ax[1].set_ylim(-20,20)
        ax[1].tick_params(axis="both",labelsize=12)


        ax[2].plot(common_timestamps[scan_mask],gain_losses,color="black")
        ax[2].set_ylim()
        ax[2].set_xlabel("Time [s]",size=14)
        ax[2].set_ylabel("Power [dB]",size=14)
        ax[2].set_title("Beam offset gain losses",size=15)
        ax[2].grid()
        ax[2].tick_params(axis="both",labelsize=12)
        
        plt.tight_layout()
        
        if savefig:
            plt.savefig(f"{out_dir_fig}/gain_losses.png",dpi=300)
            
        plt.show()
        
    # get the temperature CORRECT WITH TERMOSTAT READING for 202511 campaign
    if((campaign=="202504") & (band!="WR10")):
        print("\nAverage temperature estimated from inclinometer\n")
        avg_flight_temp = np.mean(sync_data["inclinometer"]["Temper"][mask_scan])

        if plot:
            fig = plt.figure()
            plt.plot(DH.payload.inclino.data["timestamp"][mask_scan],DH.payload.inclino.data["Temper"][mask_scan],color="black")
            plt.axhline(avg_flight_temp,color="red",label=f"T ={round(avg_flight_temp,3)}")    
            plt.xlabel("Time [s]",size=14)
            plt.ylabel(r"Temperature [$^\circ$C]",size=14)
            plt.title("Average temperature during the scan",size=15)
            plt.legend(fontsize=14)
            plt.tick_params(axis="both",labelsize=14)
            plt.tight_layout()
            
            if savefig:
                plt.savefig(f"{out_dir_fig}/Average_temp.png",dpi=300)
                
            plt.show()
    elif((campaign=="202511")&(band!="WR10")):
        avg_flight_temp = 10 # need to be adjusted, this is an estimate based on previous flight

    elif((campaign=="202511") & (band=="WR10")):
        print("\nAverage temperature estimated from LM76\n")
        avg_flight_temp = np.mean(np.asarray(sync_data["lm76"]["temperature_c"]))
        
        fig,ax = plt.subplots(ncols=2,figsize=(15,7))
        
        ax[0].plot(sync_data["lm76"]["timestamp"],sync_data["lm76"]["temperature_c"],color="black")
        ax[0].axhline(avg_flight_temp,color="red",label=f"T ={round(avg_flight_temp,3)}"+r"$^\circ$C")    
        ax[0].set_xlabel("Time [s]",size=14)
        ax[0].set_ylabel(r"Temperature [$^\circ$C]",size=14)
        ax[0].set_title("Average temperature during the scan",size=15)
        ax[0].legend(fontsize=14)
        ax[0].tick_params(axis='both', labelsize=14)

        ax[1].scatter(sync_data["lm76"]["timestamp"],sync_data["lm76"]["status_crit"],
                      label="$T_{crit}$:"+str(flight.flight_info["flight_info"]["lm76_metadata"]["tcrit"])+r"$^\circ$C")
        ax[1].scatter(sync_data["lm76"]["timestamp"],sync_data["lm76"]["status_high"],
                      label="$T_{low}$:"+str(flight.flight_info["flight_info"]["lm76_metadata"]["thigh"])+r"$^\circ$C")
        ax[1].scatter(sync_data["lm76"]["timestamp"],sync_data["lm76"]["status_low"],
                      label="$T_{low}$:"+str(flight.flight_info["flight_info"]["lm76_metadata"]["tlow"])+r"$^\circ$C")

        ax[1].set_xlabel("Time [s]",size=14)
        ax[1].set_ylabel("Status",size=14)
        ax[1].set_title("Temperature sensor status",size=15)
        ax[1].legend(fontsize=14)
        ax[1].tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()

        if savefig:
            plt.savefig(f"{out_dir_fig}/Average_temp.png",dpi=300)
                
        plt.show()

    else:
        raise ValueError("Invalid campaign or frequency band")
    
    # from source characterization in the lab:------------------------------------------------------------------------------------------------------------------------
    diode = get_item_from_payload(payload,item_category="detector")
    diode_charact = get_item_lab_charact(dB_dirpath=dB_dirpath,item=diode,meas_type="responsivity",key="processed")

    freq_key = f"{int(source_freq)} GHz"

    # interpolate the diode responsivity on the temperature reading

    c_temperature = []
    control_voltage = []
    diode_output = []
    err_diode_output = []
    diode_input_pwr = []
    err_diode_input = []
    
    for temp in diode_charact[freq_key]:
        c_temperature.append(int(temp.split(" C")[0]))
    
        diode_data = diode_charact[freq_key][temp]
        
        control_voltage.append(diode_data["Control Voltage [V]"])
        diode_output.append(diode_data["Diode output [mV]"])
        err_diode_output.append(diode_data["err diode output [mV]"])
        diode_input_pwr.append(diode_data["Diode input pwr [mW]"])
        err_diode_input.append(diode_data["err diode input [mW]"])

    # Find the common control voltages across all temperatures
    cv_common = control_voltage[0]
    for cv in control_voltage[1:]:
        cv_common = np.intersect1d(cv_common, cv)
    
    print(f"Common CV points: {len(cv_common)} "
          f"(from {cv_common.min():.2f} to {cv_common.max():.2f} V)")
    
    # Slice each temperature's data to keep only the common voltages
    control_voltage_f = filter_to_common(control_voltage,    control_voltage, cv_common)
    diode_output    = filter_to_common(diode_output,    control_voltage, cv_common)
    err_diode_output = filter_to_common(err_diode_output, control_voltage, cv_common)
    diode_input_pwr = filter_to_common(diode_input_pwr,control_voltage,cv_common)
    err_diode_input = filter_to_common(err_diode_input,control_voltage,cv_common)

    
    if np.all(np.all(control_voltage_f==cv_common,axis=1)): # check if the control voltage array is the same for each temperature:

        # sort by the temperature
        sort_temp = np.argsort(c_temperature)
        diode_output = np.asarray(diode_output)[sort_temp]
        err_diode_output = np.asarray(err_diode_output)[sort_temp]
        diode_input_pwr = np.asarray(diode_input_pwr)[sort_temp]
        err_diode_input = np.asarray(err_diode_input)[sort_temp]
        c_temperature = np.sort(c_temperature)
    
        diode_interp = RegularGridInterpolator(
                            (c_temperature,cv_common),
                            np.asarray(diode_output),
                            method="linear",
                            bounds_error=True
                        )
    
        err_diode_interp = RegularGridInterpolator(
                            (c_temperature,cv_common),
                            np.asarray(err_diode_output),
                            method="linear",
                            bounds_error=True
                        )

        input_interp = RegularGridInterpolator(
                            (c_temperature,cv_common),
                            np.asarray(diode_input_pwr),
                            method="linear",
                            bounds_error=True
                        )
    
        err_input_interp = RegularGridInterpolator(
                            (c_temperature,cv_common),
                            np.asarray(err_diode_input),
                            method="linear",
                            bounds_error=True
                        )
        
    else:
        raise ValueError("Unequal sampled control voltage, stopping")

    

    """

    temp_vals = np.asarray([float(t.split(" C")[0]) for t in diode_charact[freq_key].keys()])
    temp_key = np.asarray(list(diode_charact[freq_key].keys()))[np.where(abs(temp_vals-avg_flight_temp) == min(abs(temp_vals-avg_flight_temp)))[0][0]]

    # good diode working regions: enough snr and not saturated
    good_det_mask = (diode_charact[freq_key][temp_key]["Diode output [mV]"]>1) & (diode_charact[freq_key][temp_key]["Diode output [mV]"]<256)
    
    # get the diode lab-characterization results
    diode_resp = diode_charact[freq_key][temp_key]["Diode output [mV]"][good_det_mask]
    err_diode_resp = diode_charact[freq_key][temp_key]["err diode output [mV]"][good_det_mask]
    diode_input_pwr = diode_charact[freq_key][temp_key]["Diode input pwr [mW]"][good_det_mask]
    err_diode_in_pwr = diode_charact[freq_key][temp_key]["err diode input [mW]"][good_det_mask]

    """

    # load other components characterization
    coupler = get_item_from_payload(payload,item_category="coupler")
    coupler_transm = get_item_lab_charact(dB_dirpath=dB_dirpath,item=coupler,meas_type="transmission",key=".csv")
    coupler_coupl = get_item_lab_charact(dB_dirpath=dB_dirpath,item=coupler,meas_type="coupling_performance",key=".csv")

    #print(coupler_transm,"\n",coupler_coupl)
    
    attenuators = get_item_from_payload(payload,item_category="attenuator")

    # make a list of attenuators no matter the actual length
    attenuators_list = attenuators if isinstance(attenuators, list) else [attenuators]
    
    atten_charact = []

    for att in attenuators_list:
        atten_charact.append(get_item_lab_charact(dB_dirpath=dB_dirpath,item=att,meas_type="attenuation",key=".csv"))
        
    #print("\n",atten_charact)

    antenna_gain = get_item_lab_charact(dB_dirpath=dB_dirpath,item=antenna,meas_type="radiation_pattern",key="gain_processed")
    ant_ins_loss = get_item_lab_charact(dB_dirpath=dB_dirpath,item=antenna,meas_type="insertion_loss",key=".csv")
    ant_imp_match = get_item_lab_charact(dB_dirpath=dB_dirpath,item=antenna,meas_type="impedance_matching",key="0.h5py")

    #print("\n",antenna_gain,"\n",ant_ins_loss,"\n",ant_imp_match)

    RF_components = {
        "coupler":
        {
            "name": coupler["name"],
            "Transmission [dB]": coupler_transm["S12 avg [dB]"][coupler_transm["Frequency"]==source_freq].values[0],
            "err transmission [dB]": coupler_transm["S12 std [dB]"][coupler_transm["Frequency"]==source_freq].values[0],
            "Coupling [dB]": coupler_coupl["S12 avg [dB]"][coupler_coupl["Frequency"]==source_freq].values[0],
            "err coupling [dB]": coupler_coupl["S12 std [dB]"][coupler_coupl["Frequency"]==source_freq].values[0],
        },
        "attenuators": [
        {
            "name": att["name"],
            "Attenuation [dB]": att_charact["S12 avg [dB]"][att_charact["Frequency"]==source_freq].values[0],
            "err attenuation [dB]": att_charact["S12 std [dB]"][att_charact["Frequency"]==source_freq].values[0],
        }
        for att, att_charact in zip(attenuators_list, atten_charact)
        ],
        "antenna":
            {
                "name": antenna["name"],
                "Gain [dBi]": antenna_gain["Gain [dBi]"][antenna_gain["Frequency [GHz]"]==source_freq].values[0],
                "err gain [dB]": antenna_gain["err gain [dB]"][antenna_gain["Frequency [GHz]"]==source_freq].values[0],
                "Insertion Loss [dB]": ant_ins_loss["Insertion Loss [dB]"][ant_ins_loss["Frequency [GHz]"]==source_freq].values[0],
                "Impedance Match [dB]": ant_imp_match["fsp_impedance_kapton"]["S11 avg [dB]"][ant_imp_match["fsp_impedance_kapton"]["frequency [GHz]"]==source_freq][0],
                "err impedance match [dB]": ant_imp_match["fsp_impedance_kapton"]["S11 std [dB]"][ant_imp_match["fsp_impedance_kapton"]["frequency [GHz]"]==source_freq][0]
            },
    }

    print("\n",RF_components,"\n")
    
    # compute the power from the source --------------------------------------------------------------------------------------------------------------------------
    
    # get the adc data:
    adc_data = flight.raw_data.payload_data.adc

    scan_mask_adc = (adc_data["timestamp"]>scan_start_time) & (adc_data["timestamp"]<scan_end_time)

    adc_data = adc_data.filter(scan_mask_adc)

    if(campaign=="202504"):
        adc_data["timestamp"] = adc_data["timestamp"][adc_data["amplitude"]<5*np.std(adc_data["amplitude"])]
        adc_data["amplitude"] = adc_data["amplitude"][adc_data["amplitude"]<5*np.std(adc_data["amplitude"])]

    # compute the envelope 
    filtered_signal = medfilt(adc_data["amplitude"], kernel_size=9)

    # Parameters: adjust based on your square wave period
    if band=="WR10":
        window_size = 200  # Number of samples in one square-wave high or low region

    elif band=="WR06":
        window_size = 50

    else: 
        raise ValueError("Invalid frequency band or frequency band not yet implemented")
        
    # Initialize envelope array
    envelope = np.zeros_like(filtered_signal)
    zero_level = np.zeros_like(filtered_signal)

    # Process in blocks
    for start in range(0, len(filtered_signal), window_size):
        end = min(start + window_size, len(filtered_signal))
        block = filtered_signal[start:end]
        max_val = np.max(block)
        min_val = np.min(block)
        envelope[start:end] = max_val
        zero_level[start:end] = min_val

    if plot:
        fig = plt.figure()
        plt.plot(adc_data["timestamp"],adc_data["amplitude"],ls="--",label="raw signal")
        plt.plot(adc_data["timestamp"],filtered_signal,label="filtered signal")
        plt.plot(adc_data["timestamp"],envelope, label="envelope")
        plt.plot(adc_data["timestamp"],zero_level,color="red",label="zero level")
        #plt.axhline(0)
        plt.ylabel("Voltage [mV]",size=14)
        plt.xlabel("time [s]",size=14)
        plt.title("Diode output Voltage",size=15)
        plt.legend(fontsize=14)
        plt.tick_params(axis="both",labelsize=14)
        plt.tight_layout()

        if savefig:
            plt.savefig(f"{out_dir_fig}/ADC_tod.png",dpi=300)
        
        plt.show()

    print("\nlen None:"+str(len(envelope[np.isnan(envelope)])),"\n")

    adc_data = adc_data.filter(~np.isnan(envelope))
    
    adc_time = np.asarray(adc_data["timestamp"])
    adc_ampl = np.asarray(adc_data["amplitude"])
    adc_envl = np.asarray(envelope[~np.isnan(envelope)])

    # downsample to telescope sampling rate (200 Hz)
    fs_new = 20 # Hz
    adc_sr = 1600              # adc sampling rate
    ds_factor = int(adc_sr/fs_new)  # downsampling factor
    
    time_downsampler = adc_time[:len(adc_time)//ds_factor*ds_factor].reshape(-1, ds_factor)
    envlp_downsampler = adc_envl[:len(adc_envl)//ds_factor*ds_factor].reshape(-1, ds_factor)

    time_ds = time_downsampler.mean(axis=1)
    envlp_ds = envlp_downsampler.mean(axis=1)
    err_envlp_ds = envlp_downsampler.std(axis=1,ddof=1)/np.sqrt(ds_factor)

    # align to pils timestream
    t_start = max(time_ds[0], common_timestamps[scan_mask][0])
    t_end = min(time_ds[-1], common_timestamps[scan_mask][-1])

    t_common = np.arange(t_start, t_end, 1/fs_new)

    envlp_ds_aligned = np.interp(t_common, time_ds, envlp_ds)
    err_envlp_ds_aligned = np.interp(t_common, time_ds, err_envlp_ds)
    dist_aligned = np.interp(t_common, common_timestamps[scan_mask], dist)
    point_loss_aligned = np.interp(t_common,common_timestamps[scan_mask],gain_losses)
    temperature_aligned = np.interp(t_common,common_timestamps[scan_mask],np.asarray(sync_data["lm76"]["temperature_c"]))

    print("\nExtracted diode output voltage envelope:",envlp_ds_aligned,"\n")

    # compute the emitted power
    source_out_pwr_dBm = np.zeros(len(envlp_ds_aligned))     #np.zeros(len(adc_envl[::computation_range]))
    source_out_pwr_err_mW = np.zeros(len(envlp_ds_aligned))  #np.zeros(len(adc_envl[::computation_range]))
    
    for i in tqdm(range(len(envlp_ds_aligned))):

        if temperature_aligned[i]>10:
            temp_interp = 10
        elif temperature_aligned[i]<-10:
            temp_interp = -10
        else: 
            temp_interp = temperature_aligned[i]
        
        y_nd,sigma_y_nd = interp_error_prop.interpolate_with_errors(x_cal=diode_interp((temp_interp,cv_common)), y_cal=input_interp((temp_interp,cv_common)), 
                                                                    delta_x_cal=err_diode_interp((temp_interp,cv_common)), delta_y_cal=err_input_interp((temp_interp,cv_common)),
                                                                    x_new=envlp_ds_aligned[i], delta_x_new=err_envlp_ds_aligned[i], 
                                                                    method='linear')
        
        out_pwr, out_pwr_std = get_output_pwr_Interp1d(y_nd,
                                                       sigma_y_nd,
                                                       RF_components
                                                       )

        source_out_pwr_dBm[i] = out_pwr
        source_out_pwr_err_mW[i] = out_pwr_std

    print("\nSource output pwr [dBm]:",source_out_pwr_dBm,"\n")

    if plot:
        fig = plt.figure()
        plt.plot(t_common,dBm_to_mW(source_out_pwr_dBm),color="black")
        plt.fill_between(t_common, dBm_to_mW(source_out_pwr_dBm)-source_out_pwr_err_mW,
                         dBm_to_mW(source_out_pwr_dBm)+source_out_pwr_err_mW,alpha=0.5,color="yellow")
        plt.xlabel("Time [s]",size=14)
        plt.ylabel("Power [mW]",size=14)
        plt.title("Source output power",size=15)
        plt.tick_params(axis="both",labelsize=12)
        plt.tight_layout()

        if savefig:
            plt.savefig(f"{out_dir_fig}/Source_out_pwr.png",dpi=300)
        
        plt.show()

    # correct the power for pointing losses:
    outpwr_point_correct = source_out_pwr_dBm - point_loss_aligned
    # propagate the uncertainty
    err_outpwr_mW_point = source_out_pwr_err_mW * 10**(point_loss_aligned/10)

    # compute expected power on detectors
    expctd_det_pwr = np.zeros(len(source_out_pwr_dBm))
    err_expctd_det_pwr = np.zeros(len(source_out_pwr_dBm))
    drone_pwr = np.zeros(len(source_out_pwr_dBm))
    
    # define the bandpass:
    if(bandpass_str=="gab"):
        bandpass = 30 #GHz
    elif(bandpass_str=="SO"):
        bandpass = 26.9 # GHz
    else:
        print("Invalid bandpass string: setting the default: 30GHz")
    
    for i in range(len(source_out_pwr_dBm)):
        # source power
        det_power,std_det_pwr = get_detector_power(outpwr_point_correct[i]*u.dB, err_outpwr_mW_point[i], detector_gain, freq=freqs[band], 
                                                   optical_efficiency=1,distance=dist_aligned[i])
        expctd_det_pwr[i] = dBm_to_mW(det_power.value)*1e9 # pW #10**((det_power.value-30)/10)*1e12 # pW
        err_expctd_det_pwr[i] = std_det_pwr*1e9 # pW

        # drone power
        drone_emission = get_drone_power(temp=300, distance=dist_aligned[i], effective_diameter=0.5, beam_solid_angle=omega, 
                                         optical_efficiency=1, bandpass=bandpass)
        drone_pwr[i] = 10**((drone_emission.value-30)/10)*1e12 # pW
    
    print("\nExpected pwr [pw]:",expctd_det_pwr,"\n")

    if plot:
        plt.figure()
        plt.plot(t_common,expctd_det_pwr,color="black")
        plt.fill_between(t_common, expctd_det_pwr-err_expctd_det_pwr,expctd_det_pwr+err_expctd_det_pwr,alpha=0.5,color="yellow")
        plt.title("Expected source power on detectors",size=15)
        plt.xlabel("Time [s]",size=14)
        plt.ylabel("Power [pW]",size=14)
        plt.tick_params(axis="both",labelsize=12)
        plt.tight_layout()
        
        if savefig:
            plt.savefig(f"{out_dir_fig}/Expected_pwr.png",dpi=300)

        plt.show()

    # store the results in a pandas DataFrame

    results_df = pd.DataFrame({
                               "timestamp [s]": t_common,
                               "expected_det_power [pW]": expctd_det_pwr,
                               "err_expected_det_power [pW]": err_expctd_det_pwr,
                               "drone_power [pW]": drone_pwr,
                               "source_output_power [dBm]": outpwr_point_correct,
                               "err_output_power [mW]": err_outpwr_mW_point,
                               "distance [m]": dist_aligned,
                              })

    return results_df, fs_new, bandpass, omega

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Get power on the telescope from adc data.")
    parser.add_argument("-n","--flight-ID",type=str,help="ID of the flight")
    parser.add_argument("-t","--telescope-ID",type=str,help="Name of the targeted telescope")
    parser.add_argument("-w","--wafer-slot",type=str,help="Name of the targeted wafer-slot")
    parser.add_argument("-s","--setting-det",type=str,default="gab",help="Bandpass and beam solid angle parameters")
    parser.add_argument("-p","--plot",type=bool,default=True,help="plot the results")

    args = parser.parse_args()

    # get the date and campaign ID from the flight_ID:
    date = args.flight_ID.split("_")[0]

    # load the flight data with stout:
    stout_loader = StoutLoader()

    if(("202511" in date) | ("202512" in date)):
        camp_service = CampaignService()
        campaign_ID = "202511"
        flight_meta = stout_loader.load_single_flight(flight_name="flight_"+args.flight_ID)
        payload = camp_service.get_payload_by_id(payload_id=flight_meta["payload_id"])

        multiplier = get_item_from_payload(payload,item_category="multiplier")

        if multiplier["specifications"]["output_band"]=="WR10":
            payload_sensors = ["lm76","adc","inclinometer","gps"]
        else:
            payload_sensors = ["adc","inclinometer","gps"]

    elif(("202504" in date) | ("202505" in date)):
        campaign_ID = "202504"
        flight_meta = stout_loader._load_single_flight_from_filesystem(flight_name="flight_"+args.flight_ID)
        payload_sensors = ["adc","inclinometer","gps"]

    else:
        raise ValueError("Wrong campaign ID: it should be 202511 or 202504")

    print("date:",date,"\ncampaign_ID:",campaign_ID,"\nflight:",args.flight_ID,"\n")

    flight = Flight(flight_meta)
    flight.add_sensor_data(payload_sensors)
    flight.add_drone_data()

    out_dir = f"/home/jupyter-ludovico/optical efficiency/results/{campaign_ID}/{args.telescope_ID}/{date}/flight_{args.flight_ID}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    results_df, sampling_freq, bandpass, omega = compute_power_on_det(flight=flight,campaign=campaign_ID,payload=payload,tele=args.telescope_ID,ws=args.wafer_slot,
                                                                      out_dir=out_dir,bandpass_str=args.setting_det)
    
    outfile = f"{out_dir}/stout_expected_pwr.csv"

    header_lines = [
                    f"# Signal frequency: {flight.flight_info["flight_info"]["frequency"]} GHz",
                    f"# Sampling rate: {sampling_freq} Hz",
                    f"# Telescope bandpass: {bandpass} GHz",
                    f"# Detector beam solid angle: {omega} usr^2",
                    ]
    
    with open(outfile, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        results_df.to_csv(f, index=False)
        