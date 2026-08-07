import re
import sys
from pathlib import Path
import yaml
import h5py
import numpy as np
import pandas as pd
import scipy
import argparse

def gaus(x,amp,mean,sigma):
        return amp * np.exp(-(x - mean)**2 / (sigma**2))

def dB_to_linear(pwr_dB):
    return 10**(pwr_dB/10)

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

def load_notes(notes_path):
    with open(notes_path, "r", encoding="utf-8") as f:
        content = f.read()

    # extract the yaml bloc within: "---"
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    if not match:
        raise ValueError("No configuration block found in the notes file.")

    front_matter = yaml.safe_load(match.group(1))
    
    return front_matter["config"]

def extract_fwhm_from_dict(df,plane):

    d = df[plane]

    freqs = []
    fwhm = []
    err_fwhm = []

    for key, val in d.items():
        freq = float(key.replace(' GHz', ''))
        freqs.append(freq)
        fwhm.append(val['fwhm'])
        err_fwhm.append(val['err_fwhm'])

    # sort by freq
    freqs, fwhm, err_fwhm = zip(*sorted(zip(freqs, fwhm, err_fwhm)))
    
    return np.asarray(freqs), np.asarray(fwhm), np.asarray(err_fwhm) 
