#!/usr/bin/env python
import numpy.testing as npt
import os
from caiman.source_extraction import cnmf
from caiman.paths import caiman_datadir


def initialize_cnmf(fname, fr, decay_time, gSig, p, min_SNR, thresh_CNN_noisy, gnb, init_method,
                    init_batch, patch_size, stride, K):
    params_dict = {
        'fr': fr,
        'fnames': fname,
        'decay_time': decay_time,
        'gSig': gSig,
        'p': p,
        'motion_correct': False,
        'min_SNR': min_SNR,
        'nb': gnb,
        'init_batch': init_batch,
        'init_method': init_method,
        'rf': patch_size // 2,
        'stride': stride,
        'sniper_mode': True,
        'thresh_CNN_noisy': thresh_CNN_noisy,
        'K': K
    }
    opts = cnmf.params.CNMFParams(params_dict=params_dict)
    cnm = cnmf.online_cnmf.OnACID(params=opts)
    cnm.fit_online()
    cnm.save('test_online.hdf5')


def load_cnmf(filename):
    cnm = cnmf.online_cnmf.load_OnlineCNMF(filename)
    return cnm


def test_cnmf_estimates(cnm1, cnm2):
    npt.assert_allclose(cnm1.estimates.A.sum(), cnm2.estimates.A.sum())
    npt.assert_allclose(cnm1.estimates.C, cnm2.estimates.C)


def demo():
    fname = [os.path.join(caiman_datadir(), 'example_movies', 'demoMovie.tif')]
    fr = 10                    # frame rate (Hz)
    decay_time = .75           # approximate length of transient event in seconds
    gSig = [6, 6]              # expected half size of neurons
    p = 1                      # order of AR indicator dynamics
    min_SNR = 1                # minimum SNR for accepting candidate components
    thresh_CNN_noisy = 0.65    # CNN threshold for candidate components
    gnb = 2                    # number of background components
    init_method = 'cnmf'       # initialization method
    init_batch = 400           # number of frames for initialization
    patch_size = 32            # size of patch
    stride = 3                 # amount of overlap between patches
    K = 4                      # max number of components in each patch
    
    initialize_cnmf(fname, fr, decay_time, gSig, p, min_SNR, thresh_CNN_noisy, gnb, init_method,
                    init_batch, patch_size, stride, K)

    cnm1 = load_cnmf('test_online.hdf5')
    cnm2 = load_cnmf('test_online.hdf5')
    test_cnmf_estimates(cnm1, cnm2)


def test_onacid():
    demo()
    pass