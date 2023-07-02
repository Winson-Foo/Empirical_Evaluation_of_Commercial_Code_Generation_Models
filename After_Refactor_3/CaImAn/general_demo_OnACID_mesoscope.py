import glob
import numpy as np
import os
import logging
import matplotlib.pyplot as plt

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction import cnmf as cnmf
from caiman.utils.utils import download_demo

logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"
                           "[%(process)d] %(message)s", level=logging.INFO)

def main():
    fnames = download_demo_files()
    params = setup_parameters(fnames)
    cnm = fit_online_cnmf(params)
    plot_contours(cnm)
    view_components(cnm)
    plot_processing_time(cnm)

def download_demo_files():
    fld_name = 'Mesoscope'
    fnames = []
    fnames.append(download_demo('Tolias_mesoscope_1.hdf5', fld_name))
    fnames.append(download_demo('Tolias_mesoscope_2.hdf5', fld_name))
    fnames.append(download_demo('Tolias_mesoscope_3.hdf5', fld_name))
    logging.info(fnames)
    return fnames

def setup_parameters(fnames):
    fr = 15
    decay_time = 0.5
    gSig = (3, 3)
    p = 1
    min_SNR = 1
    ds_factor = 1
    gnb = 2
    gSig = tuple(np.ceil(np.array(gSig) / ds_factor).astype('int'))
    mot_corr = True
    pw_rigid = False
    max_shifts_online = np.ceil(10.).astype('int')
    sniper_mode = True
    rval_thr = 0.9
    init_batch = 200
    K = 2
    epochs = 1
    show_movie = False

    params_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'gSig': gSig,
        'p': p,
        'min_SNR': min_SNR,
        'rval_thr': rval_thr,
        'ds_factor': ds_factor,
        'nb': gnb,
        'motion_correct': mot_corr,
        'init_batch': init_batch,
        'init_method': 'bare',
        'normalize': True,
        'sniper_mode': sniper_mode,
        'K': K,
        'epochs': epochs,
        'max_shifts_online': max_shifts_online,
        'pw_rigid': pw_rigid,
        'dist_shape_update': True,
        'min_num_trial': 10,
        'show_movie': show_movie
    }

    return cnmf.params.CNMFParams(params_dict=params_dict)

def fit_online_cnmf(params):
    cnm = cnmf.online_cnmf.OnACID(params=params)
    cnm.fit_online()
    return cnm

def plot_contours(cnm):
    logging.info('Number of components: ' + str(cnm.estimates.A.shape[-1]))
    images = cm.load(cnm.params['fnames'])
    Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    cnm.estimates.plot_contours(img=Cn, display_numbers=False)

def view_components(cnm):
    images = cm.load(cnm.params['fnames'])
    Cn = images.local_correlations(swap_dim=False, frames_per_chunk=500)
    cnm.estimates.view_components(img=Cn)

def plot_processing_time(cnm):
    T_motion = 1e3 * np.array(cnm.t_motion)
    T_detect = 1e3 * np.array(cnm.t_detect)
    T_shapes = 1e3 * np.array(cnm.t_shapes)
    T_track = 1e3 * np.array(cnm.t_online) - T_motion - T_detect - T_shapes
    plt.figure()
    plt.stackplot(np.arange(len(T_motion)), T_motion, T_track, T_detect, T_shapes)
    plt.legend(labels=['motion', 'tracking', 'detect', 'shapes'], loc=2)
    plt.title('Processing time allocation')
    plt.xlabel('Frame #')
    plt.ylabel('Processing time [ms]')

if __name__ == "__main__":
    main()