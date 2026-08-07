import numpy as np
import astropy.units as u
import astropy.constants as const
import scipy
import argparse
from scipy.signal import medfilt
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

def get_item_from_paylaod(payload,item_category):
    """
    Returns the desired item given the flying payload object and the item category string.
    """
    
    for i in payload["items"]:
            if i["category"]==item_category:
                item = i

    return item

def get_out_pwr_coupl_Interp1d(in_pwr_diode,std_in_pwr_diode,transm, coupling):

    # estimate the output power from the directional coupler taking into account the transmissivity and the directivity of the coupler
    out_pwr_coupl_dBm = mW_to_dBm(in_pwr_diode * 10**((transm-coupling)/10))
    std_out_pwr_coupl_mW = std_in_pwr_diode * 10**((transm-coupling)/10) # assuming negligible the uncertainties on the coupling and transmission factor of the coupler
    
    return out_pwr_coupl_dBm,std_out_pwr_coupl_mW
    

def get_output_pwr_Interp1d(in_pwr_diode,std_in_pwr_diode,coupler,components,campaign,freq_idx,band):

    if band=="f090":
        
        if campaign=="202504":
            attenuation = components["Attenuator 30dB"][freq_idx]
            antenna_gain = components["Near field probe"][freq_idx]

        elif(campaign=="202511"):
            attenuation = components["Attenuator 30dB"][freq_idx] + components["Attenuator 20dB"][freq_idx]
            antenna_gain = components["15 dBi antenna"][freq_idx]

        else:
            raise ValueError("Wrong campaign ID: it should be 202511 or 202504")

    elif band=="f150":
        attenuation = components["Attenuator 30dB elmika"][freq_idx]
        antenna_gain = components["15 dBi antenna"][freq_idx]

    else:
        raise ValueError("Invalid frequency band or frequency band not yet implemented")

    out_pwr_coupl_dBm,std_out_pwr_coupl_mW = get_out_pwr_coupl_Interp1d(in_pwr_diode,std_in_pwr_diode,
                                                                        coupler["Transmission [dB]"][freq_idx], coupler["Coupling [dB]"][freq_idx])
    
    source_out_pwr_dBm = out_pwr_coupl_dBm + attenuation + antenna_gain
    source_out_pwr_std_mW = std_out_pwr_coupl_mW * 10**((attenuation + antenna_gain)/10)
    
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

    multiplier = get_item_from_paylaod(paylaod,item_category="multiplier")
    band = multiplier["specifications"]["output_band"]
    freqs[band] = source_freq

    print("\n",freqs,"\n")

    if "satp" in tele:
        print("\nTelescope name:",tele)
        # Get beam solid angle
        if band=="WR10":
            fwhm_df = pd.read_csv(f"beams/{tele}/fwhms_{tele}_{ws}.csv")
        
            if((source_freq!=90) & (source_freq!=96)):
                fwhms[band] = fwhm_df[fwhm_df["freq"]==source_freq]["fwhm"].values[0]

                # compute the beam solid angle from drone characterization
                if tele=="satp1":
                    beam_path = f"beams/{tele}/coadded_invw_fitted_full_unsmoothed_prof_T_{tele}_{int(source_freq)}_{ws}_mid.txt"
                    angle,power = np.loadtxt(beam_path,unpack=True)
                    angle_deg = angle/60
                    log_pow = 10*np.log10(power/max(power))
                elif tele =="satp3":
                    beam_path = f"beams/{tele}/coadded_unsmoothed_prof_T_{tele}_{int(source_freq)}_{ws}_mid.txt"
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
        azel_data = azel.run_analysis(tele, dji_broadcast_geod={'lat': -22.9597732, 'lon': -67.7866847, 'alt': 5173.020})
        
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
    antenna = get_item_from_paylaod(payload,item_category="antenna")
    
    if(flight.flight_info["flight_type"]=="Beam/Pol. Angle"):
        
        if band=="f090":
            
            if campaign=="202511":
                file_Hplane = "/home/jupyter-ludovico/source_beams/WR10/plots/horn_30deg/FWHM_Hplane"
                file_Eplane = "/home/jupyter-ludovico/source_beams/WR10/plots/horn_30deg/FWHM_Eplane"
                
            elif campaign=="202504":
                file_Hplane = "/home/jupyter-ludovico/source_beams/WR10/plots/nfp/FWHM_Hplane"
                file_Eplane = "/home/jupyter-ludovico/source_beams/WR10/plots/nfp/FWHM_Eplane"

            else:
                raise ValueError("Wrong campaign ID: it should be 202511 or 202504")

        elif band=="f150":
            file_Hplane = "/home/jupyter-ludovico/source_beams/WR6/plots/horn_30deg/FWHM_Hplane"
            file_Eplane = "/home/jupyter-ludovico/source_beams/WR6/plots/horn_30deg/FWHM_Eplane"

        else:
            raise ValueError("Invalid frequency band or frequency band not yet implemented")

    elif(flight_type=="Sidelobes"):
        file_Hplane = "/home/jupyter-ludovico/source_beams/WR10/plots/LPage_horn/FWHM_Hplane"
        file_Eplane = "/home/jupyter-ludovico/source_beams/WR10/plots/LPage_horn/FWHM_Eplane"

    else:
        raise ValueError("Invalid flight type")

    FWHM_H,err_FWHM_H,FWHM_E,err_FWHM_E = pointing_correction_tools.load_measured_beams(file_Hplane,file_Eplane,freq=source_freq)
    
    # --- Coordinate grid ---
    x = np.linspace(-90, 90, 501)
    y = np.linspace(-90, 90, 501)
    
    # --- 2D Gaussian beam ---
    beam_dB = 10*np.log10(pointing_correction_tools.beam_2D_gauss(x,y,FWHM_H,FWHM_E))
    beam_dB = np.maximum(beam_dB, -60)

    # interpolating the 2D linear beam to estimate values
    beam_lin_interp = scipy.interpolate.RegularGridInterpolator((y,x),pointing_correction_tools.beam_2D_gauss(x,y,FWHM_H,FWHM_E),
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
    elif((campaign=="202511")&(band!="f090")):
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
    
    # load diode-detector characterization
    diode_charact_file = f'lab_characterization/{band}/diode_responsivity/eravant_diode_{int(source_freq)}GHz_temp.pickle'
    print("\n",diode_charact_file,"\n")
    with open(diode_charact_file, 'rb') as handle:
        diode_dict = pickle.load(handle)
        
    temp_idx = np.where(abs(diode_dict["temp [°C]"]-avg_flight_temp) == min(abs(diode_dict["temp [°C]"]-avg_flight_temp)))[0][0]

    # get the diode lab-characterization results
    mask_range = diode_dict["mask_range"][temp_idx]
    diode_resp = diode_dict["diode_response [mV]"][temp_idx][mask_range]
    err_diode_resp = diode_dict["sigma_diode_resp [mV]"][temp_idx][mask_range]
    diode_input_pwr = diode_dict["input_pwr [mW]"][temp_idx][mask_range]
    err_diode_in_pwr = diode_dict["sigma_input_pwr [mW]"][temp_idx][mask_range]

    # load other components characterization
    if(band=="WR10"):
        coupler = pd.read_csv(f"lab_characterization/{band}/20dBcoupler.csv")
    elif(band=="WR6"):
        coupler = pd.read_csv(f"lab_characterization/{band}/20dB_XGuide.csv")

    freq_idx = coupler[coupler["Frequency [GHz]"] == source_freq].index[0]

    components = pd.read_csv(f"lab_characterization/{band}/{band}_components.csv")

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
    if band=="f090":
        window_size = 200  # Number of samples in one square-wave high or low region

    elif band=="f150":
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

    print("\nExtracted diode output voltage envelope:",envlp_ds,"\n")

    # compute the emitted power
    source_out_pwr_dBm = np.zeros(len(envlp_ds))     #np.zeros(len(adc_envl[::computation_range]))
    source_out_pwr_err_mW = np.zeros(len(envlp_ds))  #np.zeros(len(adc_envl[::computation_range]))
    
    for i in tqdm(range(len(envlp_ds))):
        y_nd,sigma_y_nd = interp_error_prop.interpolate_with_errors(x_cal=diode_resp, y_cal=diode_input_pwr, 
                                                                    delta_x_cal=err_diode_resp, delta_y_cal=err_diode_in_pwr,
                                                                    x_new=envlp_ds[i], delta_x_new=err_envlp_ds[i], 
                                                                    method='linear')
        
        out_pwr, out_pwr_std = get_output_pwr_Interp1d(y_nd,
                                                       sigma_y_nd,
                                                       coupler,
                                                       components,
                                                       campaign,
                                                       freq_idx,
                                                       band
                                                       )

        source_out_pwr_dBm[i] = out_pwr
        source_out_pwr_err_mW[i] = out_pwr_std

    print("\nSource output pwr [dBm]:",source_out_pwr_dBm,"\n")

    if plot:
        fig = plt.figure()
        plt.plot(time_ds,dBm_to_mW(source_out_pwr_dBm),color="black")
        plt.fill_between(time_ds, dBm_to_mW(source_out_pwr_dBm)-source_out_pwr_err_mW,
                         dBm_to_mW(source_out_pwr_dBm)+source_out_pwr_err_mW,alpha=0.5,color="yellow")
        plt.xlabel("Time [s]",size=14)
        plt.ylabel("Power [mW]",size=14)
        plt.title("Source output power",size=15)
        plt.tick_params(axis="both",labelsize=12)
        plt.tight_layout()

        if savefig:
            plt.savefig(f"{out_dir_fig}/Source_out_pwr.png",dpi=300)
        
        plt.show()

    # align pils timestream to compute the fspl and pointing correction
    t_start = max(time_ds[0], common_timestamps[scan_mask][0])
    t_end = min(time_ds[-1], common_timestamps[scan_mask][-1])

    t_common = np.arange(t_start, t_end, 1/fs_new)

    outpwr_dBm_aligned = np.interp(t_common, time_ds, source_out_pwr_dBm)
    err_outpwr_mW_aligned = np.interp(t_common, time_ds, source_out_pwr_err_mW)
    dist_aligned = np.interp(t_common, common_timestamps[scan_mask], dist)
    point_loss_aligned = np.interp(t_common,common_timestamps[scan_mask],gain_losses)

    # correct the power for pointing losses:
    outpwr_point_correct = outpwr_dBm_aligned - point_loss_aligned
    # propagate the uncertainty
    err_outpwr_mW_point = err_outpwr_mW_aligned * 10**(point_loss_aligned/10)

    # compute expected power on detectors
    expctd_det_pwr = np.zeros(len(outpwr_dBm_aligned))
    err_expctd_det_pwr = np.zeros(len(outpwr_dBm_aligned))
    drone_pwr = np.zeros(len(outpwr_dBm_aligned))
    
    # define the bandpass:
    if(bandpass_str=="gab"):
        bandpass = 30 #GHz
    elif(bandpass_str=="SO"):
        bandpass = 26.9 # GHz
    else:
        print("Invalid bandpass string: setting the default: 30GHz")
    
    for i in range(len(outpwr_dBm_aligned)):
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
                               "source_output_power [dBm]": outpwr_dBm_aligned,
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

        multiplier = get_item_from_paylaod(paylaod,item_category="multiplier")

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

    out_dir = f"results/{campaign_ID}/{args.telescope_ID}/{date}/flight_{args.flight_ID}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    results_df, sampling_freq, bandpass, omega = compute_power_on_det(flight=flight,campaign=campaign_ID,payload=payload,tele=args.telescope_ID,ws=args.wafer_slot,
                                                                      out_dir=out_dir,bandpass_str=args.setting_det)
    
    outfile = f"{out_dir}/expected_pwr.csv"

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
        