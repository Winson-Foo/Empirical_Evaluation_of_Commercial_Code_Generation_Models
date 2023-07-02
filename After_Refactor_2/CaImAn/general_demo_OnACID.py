#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import os

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.paths import caiman_datadir


def setup_logger():
    logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"
                        "[%(process)d] %(message)s",
                        level=logging.WARNING)


def load_data():
    fname = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]
    return fname


def set_parameters():
    fr = 10  # frame rate (Hz)
    decay_time = 0.75  # approximate length of transient event in seconds
    gSig = [6, 6]  # expected half size of neurons
    p = 1  # order of AR indicator dynamics
    min_SNR = 1  # minimum SNR for accepting candidate components
    thresh_CNN_noisy = 0.65  # CNN threshold for candidate components
    gnb = 2  # number of background components
    init_method = 'cnmf'  # initialization method
    init_batch = 400  # number of frames for initialization
    patch_size = 32  # size of patch
    stride = 3  # amount of overlap between patches
    K = 4  # max number of components in each patch

    params_dict = {'fr': fr,
                   'fnames': load_data(),
                   'decay_time': decay_time,
                   'gSig': gSig,
                   'p': p,
                   'min_SNR': min_SNR,
                   'nb': gnb,
                   'init_batch': init_batch,
                   'init_method': init_method,
                   'rf': patch_size//2,
                   'stride': stride,
                   'sniper_mode': True,
                   'thresh_CNN_noisy': thresh_CNN_noisy,
                   'K': K}
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    return opts


def fit_online(opts):
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()
    return cnm


def plot_contours(cnm):
    logging.info('Number of components:' + str(cnm.estimates.A.shape[-1]))
    Cn = cm.load(load_data()[0], subindices=slice(0, 500)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours(img=Cn)


def apply_CNN_classifier(opts):
    use_CNN = True
    if use_CNN:
        opts.set('quality', {'min_cnn_thr': 0.05})
        cnm.estimates.evaluate_components_CNN(opts)
        cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)


def view_results(cnm):
    Cn = cm.load(load_data()[0], subindices=slice(0, 500)).local_correlations(swap_dim=False)
    cnm.estimates.view_components(img=Cn, idx=cnm.estimates.idx_components)


def main():
    setup_logger()
    opts = set_parameters()
    cnm = fit_online(opts)
    plot_contours(cnm)
    apply_CNN_classifier(opts)
    view_results(cnm)


# This is to mask the differences between running this demo in Spyder
# versus from the CLI
if __name__ == "__main__":
    main()