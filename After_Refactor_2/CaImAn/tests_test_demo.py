#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
import os
import caiman as cm
import caiman.paths
from caiman.source_extraction import cnmf


def load_movie_and_memorymap():
    fname_new = cm.save_memmap([os.path.join(caiman.paths.caiman_datadir(), 'example_movies', 'demoMovie.tif')],
                               base_name='Yr',
                               order='C')
    Yr, dims, T = cm.load_memmap(fname_new)
    return Yr, dims, T


def initialize_cnmf(n_processes, dview, p):
    cnm = cnmf.CNMF(n_processes,
                    method_init='greedy_roi',
                    k=30,
                    gSig=[4, 4],
                    merge_thresh=.8,
                    p=p,
                    dview=dview,
                    Ain=None,
                    method_deconvolution='oasis')
    return cnm


def fit_model(cnm, Yr):
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm = cnm.fit(images)
    return cnm


def save_model(cnm, filename):
    cnm.save(filename)


def load_model(filename):
    cnm = cnmf.cnmf.load_CNMF(filename)
    return cnm


def test_model_similarity(cnm1, cnm2):
    npt.assert_allclose(cnm1.estimates.A.sum(), cnm2.estimates.A.sum())
    npt.assert_allclose(cnm1.estimates.C, cnm2.estimates.C)


def demo(parallel=False):
    p = 2
    if parallel:
        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    else:
        n_processes, dview = 2, None

    Yr, dims, T = load_movie_and_memorymap()
    cnm = initialize_cnmf(n_processes, dview, p)
    cnm = fit_model(cnm, Yr)

    if parallel:
        cm.cluster.stop_server(dview=dview)

    save_model(cnm, 'test_file.hdf5')
    cnm2 = load_model('test_file.hdf5')
    test_model_similarity(cnm, cnm2)

    try:
        dview.terminate()
    except:
        pass


def test_single_thread():
    demo()


def test_parallel():
    demo(True)