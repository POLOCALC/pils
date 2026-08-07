import re
import sys
from pathlib import Path
import yaml
import h5py
import numpy as np
import pandas as pd
import scipy
import argparse
import copy

import sys
sys.path.insert(0, "/home/jupyter-ludovico/stout")

from stout.services.inventory.service import InventoryService
from stout.services.measurements.service import MeasurementService

import utils

class Beam_Processing:
    
    def __init__(self, dirpath, date):

        path = Path(dirpath)
        
        raw_files = sorted(p for p in path.iterdir() if p.is_file() and p.name.lower().endswith("_raw.h5py") and date in p.name.lower())
        notes_file = sorted(p for p in path.iterdir() if p.is_file() and p.name.lower().endswith(".md") and date in p.name.lower())

        self.dirpath = path
        self.date = date
        self.raw_file = raw_files[0]
        self.notes_file = notes_file[0]
        
        self.beam_df = utils.load_data(self.raw_file)

        config = utils.load_notes(self.notes_file)

        test_antenna_id = config["components"]["receiving_antenna"]["item_id"]
        launch_antenna_id = config["components"]["launching_antenna"]["item_id"]

        self.grid_status = config["auxiliary_components"]["grid_status"]

        inv_service = InventoryService()
        self.test_antenna = inv_service.get_item_by_id(item_id=test_antenna_id)
        self.launch_antenna = inv_service.get_item_by_id(item_id=launch_antenna_id)
        self.nom_gain = self.test_antenna["specifications"]["gain"]
        
        self.fwhm_df = None
        self.gain_df = None

        
    def get_single_freq_beam(self,freq,plane="all",normalize=True):

        if plane=="all":
            single_freq_beam = {}
            for cut in self.beam_df.keys():
                single_freq_beam.update({cut: {}})
                
                for pol in self.beam_df[cut].keys():
                    single_freq_beam[cut].update({pol: {}})
                    
                    for k in self.beam_df[cut][pol]:
                        if normalize:

                            if k=="S12(DB)":

                                S12 = self.beam_df[cut][pol]["S12(DB)"][self.beam_df[cut][pol]["Freq(GHz)"]==freq]
                                ang = self.beam_df[cut][pol]["Ang(DEG)"][self.beam_df[cut][pol]["Freq(GHz)"]==freq]

                                # normalize to the copol on axis
                                S12_norm = S12 - self.beam_df[cut]["copol"]["S12(DB)"][self.beam_df[cut][pol]["Freq(GHz)"]==freq][ang==0]

                                single_freq_beam[cut][pol].update({"S12(DB)": S12_norm})

                            else:
                                single_freq_beam[cut][pol].update({k: self.beam_df[cut][pol][k][self.beam_df[cut][pol]["Freq(GHz)"]==freq]})

        else:
            single_freq_beam = {plane:{}}

            for pol in self.beam_df[plane].keys():
                single_freq_beam[plane].update({pol: {}})
                    
                for k in self.beam_df[plane][pol]:
                    if normalize:

                        if k=="S12(DB)":

                            S12 = self.beam_df[plane][pol]["S12(DB)"][self.beam_df[plane][pol]["Freq(GHz)"]==freq]
                            ang = self.beam_df[plane][pol]["Ang(DEG)"][self.beam_df[plane][pol]["Freq(GHz)"]==freq]

                            # normalize to the copol on axis
                            S12_norm = S12 - self.beam_df[plane]["copol"]["S12(DB)"][self.beam_df[plane][pol]["Freq(GHz)"]==freq][ang==0]

                            single_freq_beam[plane][pol].update({"S12(DB)": S12_norm})

                        else:
                            single_freq_beam[plane][pol].update({k: self.beam_df[plane][pol][k][self.beam_df[plane][pol]["Freq(GHz)"]==freq]})

        return single_freq_beam
        
        
    def get_fwhm(self,n_fwhm=1):

        expct_fwhm = np.rad2deg(np.sqrt(4*np.pi/(1.133*10**(self.nom_gain/10))))
        
        fwhm_df = {}

        # get the fwhm for each measured cut:
        for cut in self.beam_df.keys():

            fwhm_df.update({cut: {}})

            # single freq analysis:
            init_params = [1,0.0,2]

            for f in np.unique(self.beam_df[cut]["copol"]["Freq(GHz)"]):
                
                single_freq_beam = self.get_single_freq_beam(freq=f,plane=cut)
                copol_sf = single_freq_beam[cut]["copol"]

                ang_mask = (copol_sf["Ang(DEG)"]>-expct_fwhm*n_fwhm) & (copol_sf["Ang(DEG)"]<expct_fwhm*n_fwhm)

                # fit with a Gaussian
                popt,pcov = scipy.optimize.curve_fit(utils.gaus, copol_sf["Ang(DEG)"][ang_mask], utils.dB_to_linear(copol_sf["S12(DB)"][ang_mask]),p0=init_params)

                sigma = popt[2]
                err_sigma = np.sqrt(pcov[2][2])
                
                fwhm = 2*np.sqrt(np.log(2))*sigma
                err_fwhm = np.sqrt(np.log(2)/(sigma))*err_sigma

                fwhm_df[cut].update({f"{f} GHz": {"fwhm": fwhm,
                                                  "err_fwhm": err_fwhm}})

        self.fwhm_df = fwhm_df

    def compute_gain(self):

        if self.fwhm_df is None:
            self.get_fwhm()

        if "Hplane" in self.fwhm_df.keys() and "Eplane" in self.fwhm_df.keys():
            print("Gain computed from Hplane and Eplane")

            freq_H, fwhm_H, err_fwhm_H = utils.extract_fwhm_from_dict(self.fwhm_df,plane="Hplane")
            freq_E, fwhm_E, err_fwhm_E = utils.extract_fwhm_from_dict(self.fwhm_df,plane="Eplane")

            if np.all(freq_H==freq_E):
                freq_array = copy.copy(freq_H)
                
                # compute gain
                Omega = 1.13*np.deg2rad(fwhm_H)*np.deg2rad(fwhm_E)
                err_Omega = 1.13*np.sqrt((np.deg2rad(fwhm_H)*np.deg2rad(err_fwhm_H))**2 + (np.deg2rad(fwhm_E)*np.deg2rad(err_fwhm_E))**2)

                lin_gain = 4*np.pi/Omega
                gain = 10*np.log10(lin_gain)
                err_gain = 4.343 * err_Omega/Omega

            else:
                raise ValueError("Frequency array of the two planes need to be equal.")

        elif "Dplane" in self.fwhm_df.keys():
            print("Computing gain from Dplane")

            freq_array = copy.copy(freq_D)

            freq_D, fwhm_D, err_fwhm_D = utils.extract_fwhm_from_dict(self.fwhm_df,plane="Dplane")

            Omega = 1.13*np.deg2rad(freq_D)**2
            err_Omega = 1.13*np.sqrt(2*(np.deg2rad(fwhm_D)*np.deg2rad(err_fwhm_D))**2)

            lin_gain = 4*np.pi/Omega
            gain = 10*np.log10(lin_gain)
            err_gain = 4.343 * err_Omega/Omega

        else:
            raise ValueError("No valid planes found to compute the gain.")

        self.gain_df = {"Frequency [GHz]": freq_array,
                        "Gain [dBi]": gain,
                        "err gain [dB]": err_gain
                       }

    def save_processed_data(self):
        
        if self.gain_df is None:
            self.compute_gain()

        try:

            outfile_fwhm = self.dirpath / f"{self.date}_fwhm_processed.hdf5"

            with h5py.File(outfile_fwhm, "w") as of:
                for cut in self.fwhm_df.keys():
                    # plane cut group:
                    cut_grp = of.create_group(cut)

                    freq,fwhm,err_fwhm = utils.extract_fwhm_from_dict(self.fwhm_df,plane=cut)

                    cut_grp.create_dataset("Frequency [GHz]",data=freq)
                    cut_grp.create_dataset("FWHM [deg]",data=fwhm)
                    cut_grp.create_dataset("err FWHM [deg]",data=err_fwhm)

            print(f"\nFWHM processed data saved to: {outfile_fwhm}")

            outfile_gain = self.dirpath / f"{self.date}_gain_processed.csv"

            pd_gain_df = pd.DataFrame(self.gain_df)
            pd_gain_df.to_csv(outfile_gain,index=False)

            print(f"\nGain processed data saved to: {outfile_gain}")

        except Exception as e:
            raise ValueError(f"Error in saving the processed data: {e}")
            
                


