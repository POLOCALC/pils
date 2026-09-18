import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tools import dB_to_linear,linear_to_dB

class VNA_data:

    def __init__(self,datapath):

        # Load data given the datapath
        freq, self.S11, self.S12, self.S21, self.S22 = np.loadtxt(datapath,skiprows=8,usecols=(0,1,3,5,7),unpack=True)
        self.freq = freq*1e-9
        
        self.freq_resolution = np.mean(np.diff(self.freq))
        self.n_points = len(self.freq)
        
        self.S_correction = False

    def correct_S_params(self):

        self.S_correction = True

        # correction for the reflection coefficients:
        S11_linear,S21_linear,S12_linear,S22_linear = dB_to_linear(self.S11),dB_to_linear(self.S21),dB_to_linear(self.S12),dB_to_linear(self.S22)
    
        self.S12_corrected = linear_to_dB(S12_linear/(1-S22_linear))
        self.S21_corrected = linear_to_dB(S21_linear/(1-S11_linear))

    def plot_S_params(self):

        fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(12,10))

        ax[0,0].plot(self.freq,self.S11)
        ax[0,1].plot(self.freq,self.S12,label="Original")
        ax[1,0].plot(self.freq,self.S21,label="Original")
        ax[1,1].plot(self.freq,self.S22)

        ax[0,0].set_title("S11")
        ax[0,1].set_title("S12")
        ax[1,0].set_title("S21")
        ax[1,1].set_title("S22")

        ax[0,0].set_xlabel("Frequency [GHz]")
        ax[0,0].set_ylabel("Power [dB]")
        ax[0,1].set_xlabel("Frequency [GHz]")
        ax[0,1].set_ylabel("Power [dB]")
        ax[1,0].set_xlabel("Frequency [GHz]")
        ax[1,0].set_ylabel("Power [dB]")
        ax[1,1].set_xlabel("Frequency [GHz]")
        ax[1,1].set_ylabel("Power [dB]")

        if self.S_correction:
            ax[0,1].plot(self.freq,self.S12,label="Corrected")
            ax[1,0].plot(self.freq,self.S21,label="Corrected")
            ax[0,1].legend()
            ax[1,0].legend()
        
        #plt.tight_layout()
        fig.suptitle("S params")
        plt.show()

        if self.S_correction:
            plt.figure()
            plt.plot(self.freq,self.S12-self.S21,label="Original",alpha=0.5
                )
            plt.plot(self.freq,self.S12_corrected-self.S21_corrected,label="Reflections corrected",alpha=0.5)
            plt.title("S12-S22 Corrected vs Original")
            plt.legend()
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("S12 S21 difference [dB]")
            plt.show()

    def single_freq_analysis(self,S12,S21,interest_freqs,freq_interval=0.1):

        interest_freqs = np.atleast_1d(interest_freqs)

        S11_values = {}
        S12_values = {}
        S21_values = {}
        S22_values = {}
        
        for f in interest_freqs:
            freq_mask = (self.freq>(f-freq_interval)) & (self.freq<(f+freq_interval))

            S11_values.update({f"{f} GHz": {"avg": np.mean(self.S11[freq_mask]),
                                            "std": np.std(self.S11[freq_mask])}
                              })
            
            S12_values.update({f"{f} GHz": {"avg": np.mean(S12[freq_mask]),
                                            "std": np.std(S12[freq_mask])}
                              })

            S21_values.update({f"{f} GHz": {"avg": np.mean(S21[freq_mask]),
                                            "std": np.std(S21[freq_mask])}
                              })

            S22_values.update({f"{f} GHz": {"avg": np.mean(self.S22[freq_mask]),
                                            "std": np.std(self.S22[freq_mask])}
                              })

        return S11_values,S12_values,S21_values,S22_values

    def run_single_freqs(self,interest_freqs,method="original",freq_interval=0.1):

        if method=="corrected":
            if not self.S_correction:
                raise ValueError("S parameters must be corrected before using the 'corrected' method")

            else:
                S11_values,S12_values,S21_values,S22_values = self.single_freq_analysis(self.S12_corrected,self.S21_corrected,
                                                                  interest_freqs=interest_freqs,freq_interval=freq_interval)

        elif method=="original":
            S11_values,S12_values,S21_values,S22_values = self.single_freq_analysis(self.S12,self.S21,
                                                              interest_freqs=interest_freqs,freq_interval=freq_interval)

        else:
            raise ValueError("Invalid method: method must be 'corrected' or 'original'")

        self.single_freqs_df = pd.DataFrame({"Frequency": np.atleast_1d(interest_freqs),
                                             "S11 avg [dB]": np.asarray([S11_values[vals]["avg"] for vals in S11_values]),
                                             "S11 std [dB]": np.asarray([S11_values[vals]["std"] for vals in S11_values]),
                                             "S12 avg [dB]": np.asarray([S12_values[vals]["avg"] for vals in S12_values]),
                                             "S12 std [dB]": np.asarray([S12_values[vals]["std"] for vals in S12_values]),
                                             "S21 avg [dB]": np.asarray([S21_values[vals]["avg"] for vals in S21_values]),
                                             "S21 std [dB]": np.asarray([S21_values[vals]["std"] for vals in S21_values]),
                                             "S22 avg [dB]": np.asarray([S22_values[vals]["avg"] for vals in S22_values]),
                                             "S22 std [dB]": np.asarray([S22_values[vals]["std"] for vals in S22_values]),
                                            })

    def save_to_csv(self,outpath):
        self.single_freqs_df.to_csv(outpath,index=False)

        


            
            
                            
 


        