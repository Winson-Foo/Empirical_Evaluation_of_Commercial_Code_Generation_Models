import logging
from typing import List
import numpy as np
from . import atm
from . import spikepursuit

try:
    profile
except NameError:
    def profile(a): return a


class VOLPY:
    """Spike Detection in Voltage Imaging
    
    The general file class which is used to find spikes of voltage imaging.
    Its architecture is similar to the one of scikit-learn calling the function fit
    to run everything which is part of the structure of the class.
    The output will be recorded in self.estimates.
    In order to use VolPy within CaImAn, you must install Keras into your conda environment. 
    You can do this by activating your environment, and then issuing the command 
    "conda install -c conda-forge keras".
    """
    def __init__(
        self, 
        n_processes: int, 
        dview=None, 
        template_size: float = 0.02, 
        context_size: int = 35, 
        censor_size: int = 12, 
        visualize_ROI: bool = False, 
        flip_signal: bool = True, 
        hp_freq_pb: float = 1/3, 
        nPC_bg: int = 8, 
        ridge_bg: float = 0.01,  
        hp_freq: float = 1, 
        clip: int = 100, 
        threshold_method: str = 'adaptive_threshold', 
        min_spikes: int = 10, 
        pnorm: float = 0.5, 
        threshold: float = 3, 
        sigmas: np.ndarray = np.array([1, 1.5, 2]), 
        n_iter: int = 2, 
        weight_update: str = 'ridge', 
        do_plot: bool = False, 
        do_cross_val: bool = False, 
        sub_freq: float = 20, 
        method: str = 'spikepursuit', 
        superfactor: int = 10, 
        params=None
    ):
        """
        Args:
            n_processes: number of processes used 
            dview: Direct View object for parallelization purposes when using ipyparallel
            template_size: template_size, # half size of the window length for spike templates, default is 20 ms 
            context_size: number of pixels surrounding the ROI to use as context
            censor_size: number of pixels surrounding the ROI to censor from the background PCA; roughly
                the spatial scale of scattered/dendritic neural signals, in pixels
            flip_signal: whether to flip signal upside down for spike detection 
                True for voltron, False for others
            hp_freq_pb: high-pass frequency for removing photobleaching    
            nPC_bg: number of principal components used for background subtraction
            ridge_bg: regularization strength for ridge regression in background removal 
            hp_freq: high-pass cutoff frequency to filter the signal after computing the trace
            clip: maximum number of spikes for producing templates
            threshold_method: adaptive_threshold or simple method for thresholding signals
                adaptive_threshold method threshold based on estimated peak distribution
                simple method threshold based on estimated noise level 
            min_spikes: minimal number of spikes to be detected
            pnorm: a variable decides spike count chosen for adaptive threshold method
            threshold: threshold for spike detection in simple threshold method 
                The real threshold is the value multiplied by the estimated noise level
            sigmas: spatial smoothing radius imposed on high-pass filtered 
                movie only for finding weights
            n_iter: number of iterations alternating between estimating spike times
                and spatial filters
            weight_update: ridge or NMF for weight update
            do_plot: if True, plot trace of signals and spiketimes, 
                peak triggered average, histogram of heights in the last iteration
            do_cross_val: whether to use cross validation to optimize regression regularization parameters
            sub_freq: frequency for subthreshold extraction
            method: spikepursuit or atm method
            superfactor: used in atm method for regression
            params: parameter object
        """
        if params is None:
            logging.warning("Parameters are not set from volparams")
            raise Exception('Parameters are not set')
        else:
            self.params = params

        self.estimates = {}

    def fit(self, n_processes: int = None, dview=None):
        """Run the volspike function to detect spikes and save the result
        into self.estimates
        """
        results = []
        fnames = self.params.data['fnames']
        fr = self.params.data['fr']

        if self.params.volspike['method'] == 'spikepursuit':
            volspike = spikepursuit.volspike
        elif self.params.volspike['method'] == 'atm':
            volspike = atm.volspike    
   
        N = len(self.params.data['index'])
        times = int(np.ceil(N / n_processes))
        
        for j in range(times):
            if j < (times - 1):
                li = [k for k in range(j * n_processes, (j + 1) * n_processes)]
            else:
                li = [k for k in range(j * n_processes, N)]
            
            args_in = []
            for i in li:
                idx = self.params.data['index'][i]
                ROIs = self.params.data['ROIs'][idx]
                if self.params.data['weights'] is None:
                    weights = None
                else:
                    weights = self.params.data['weights'][i]
                args_in.append([fnames, fr, idx, ROIs, weights, self.params.volspike])

            if 'multiprocessing' in str(type(dview)):
                results_part = dview.map_async(volspike, args_in).get(4294967)
            elif dview is not None:
                results_part = dview.map_sync(volspike, args_in)
            else:
                results_part = list(map(volspike, args_in))
            results = results + results_part
        
        for i in results[0].keys():
            try:
                self.estimates[i] = np.array([results[j][i] for j in range(N)])
            except:
                self.estimates[i] = [results[j][i] for j in range(N)]
                
        return self