import numpy as np

def dB_to_linear(pow_dB):
    return 10**(pow_dB/10)

def linear_to_dB(pow_linear):
    return 10*np.log10(pow_linear)