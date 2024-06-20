#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import glob
import logging
import numpy as np
import os

import caiman as cm
from caiman.paths import caiman_datadir
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.summary_images import local_correlations_movie_offline

logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"
                           "[%(process)d] %(message)s",
                    level=logging.WARNING)


def setup_cluster():
    try:
        cv2.setNumThreads(0)
    except:
        pass

    try:
        if __IPYTHON__:
            get_ipython().magic('load_ext autoreload')
            get_ipython().magic('autoreload 2')
    except NameError:
        pass

    return cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)


def set_up_parameters(is_patches):
    fnames = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]
    fr = 10
    decay_time = 5.0

    if is_patches:
        rf = 10
        stride = 4
        K = 4
    else:
        rf = None
        stride = None
        K = 30

    gSig = [6, 6]
    merge_thresh = 0.80
    p = 2
    gnb = 2

    params_dict = {'fnames': fnames,
                   'fr': fr,
                   'decay_time': decay_time,
                   'rf': rf,
                   'stride': stride,
                   'K': K,
                   'gSig': gSig,
                   'merge_thr': merge_thresh,
                   'p': p,
                   'nb': gnb}

    opts = params.CNMFParams(params_dict=params_dict)
    return opts


def run_caiman_batch(n_processes, params, dview):
    cnm = cnmf.CNMF(n_processes, params=params, dview=dview)
    cnm = cnm.fit_file()
    return cnm


def evaluate_components(cnm2, images, dview):
    min_SNR = 2
    rval_thr = 0.85
    use_cnn = True
    min_cnn_thr = 0.99
    cnn_lowest = 0.1

    cnm2.params.set('quality', {'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': use_cnn,
                                'min_cnn_thr': min_cnn_thr,
                                'cnn_lowest': cnn_lowest})

    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)


def save_results(cnm2, Cn):
    cnm2.estimates.Cn = Cn
    cnm2.save(cnm2.mmap_file[:-4] + 'hdf5')


def main():
    is_patches = True
    
    c, dview, n_processes = setup_cluster()
    opts = set_up_parameters(is_patches)
    cnm = run_caiman_batch(n_processes, params=opts, dview=dview)

    Cns = local_correlations_movie_offline(fnames[0],
                                           remove_baseline=True,
                                           swap_dim=False, window=1000, stride=1000,
                                           winSize_baseline=100, quantil_min_baseline=10,
                                           dview=dview)
    Cn = Cns.max(axis=0)
    cnm.estimates.plot_contours(img=Cn)

    Yr, dims, T = cm.load_memmap(cnm.mmap_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    cnm2 = cnm.refit(images, dview=dview)
    evaluate_components(cnm2, images, dview)

    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
    cnm2.estimates.view_components(images, idx=cnm2.estimates.idx_components, img=Cn)

    save_results(cnm2, Cn)
    cnm2.estimates.play_movie(images, magnification=4)

    cm.stop_server(dview=dview)
    log_files = glob.glob('Yr*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)


if __name__ == "__main__":
    main()