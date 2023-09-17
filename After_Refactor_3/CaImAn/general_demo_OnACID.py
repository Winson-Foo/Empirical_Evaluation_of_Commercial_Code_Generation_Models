#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import os

import caiman as cm
from caiman.source_extraction import cnmf as cnmf
from caiman.paths import caiman_datadir

class Config:
    def __init__(self):
        self.fr = 10
        self.decay_time = 0.75
        self.gSig = [6, 6]
        self.p = 1
        self.min_SNR = 1
        self.thresh_CNN_noisy = 0.65
        self.gnb = 2
        self.init_method = 'cnmf'
        self.init_batch = 400
        self.patch_size = 32
        self.stride = 3
        self.K = 4

def setup_logger():
    logging.basicConfig(
        format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"
               "[%(process)d] %(message)s",
        level=logging.WARNING
        # filename="/tmp/caiman.log"
    )

def load_data():
    fname = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]
    return fname

def setup_params(fname, config):
    params_dict = {
        'fr': config.fr,
        'fnames': fname,
        'decay_time': config.decay_time,
        'gSig': config.gSig,
        'p': config.p,
        'min_SNR': config.min_SNR,
        'nb': config.gnb,
        'init_batch': config.init_batch,
        'init_method': config.init_method,
        'rf': config.patch_size // 2,
        'stride': config.stride,
        'sniper_mode': True,
        'thresh_CNN_noisy': config.thresh_CNN_noisy,
        'K': config.K
    }
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    return opts

def fit_online(opts):
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()
    return cnm

def plot_contours(cnm):
    logging.info('Number of components:' + str(cnm.estimates.A.shape[-1]))
    Cn = cm.load(fname[0], subindices=slice(0, 500)).local_correlations(swap_dim=False)
    cnm.estimates.plot_contours(img=Cn)

def evaluate_components(cnm):
    use_CNN = True
    if use_CNN:
        opts.set('quality', {'min_cnn_thr': 0.05})
        cnm.estimates.evaluate_components_CNN(opts)
        cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)

def view_components(cnm):
    cnm.estimates.view_components(img=Cn, idx=cnm.estimates.idx_components)

def main():
    setup_logger()
    fname = load_data()
    config = Config()
    opts = setup_params(fname, config)
    cnm = fit_online(opts)
    plot_contours(cnm)
    evaluate_components(cnm)
    view_components(cnm)

if __name__ == "__main__":
    main()