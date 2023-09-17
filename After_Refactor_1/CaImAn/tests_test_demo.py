#!/usr/bin/env python

import numpy as np
import os
import caiman as cm
import caiman.paths
from caiman.source_extraction import cnmf
import numpy.testing as npt

def load_movie():
    """
    Loads the movie and memory map it.
    Returns:
        Yr (ndarray): Flattened movie frames.
        dims (tuple): Dimensions of the movie frames.
        T (int): Number of frames in the movie.
    """
    fname_new = cm.save_memmap([os.path.join(caiman.paths.caiman_datadir(), 'example_movies', 'demoMovie.tif')],
                               base_name='Yr',
                               order='C')
    Yr, dims, T = cm.load_memmap(fname_new)
    return Yr, dims, T

def initialize_cnmf(n_processes, dview):
    """
    Initializes the CNMF object with specified parameters.
    Args:
        n_processes (int): Number of processes.
        dview (object): Distributed view object.
    Returns:
        cnm (CNMF object): Initialized CNMF object.
    """
    p = 2 # order of the AR model (in general 1 or 2)
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

def fit_cnmf(cnm, images):
    """
    Fits the CNMF model to the given images.
    Args:
        cnm (CNMF object): Initialized CNMF object.
        images (ndarray): Flattened movie frames.
    Returns:
        cnm (CNMF object): Fitted CNMF object.
    """
    cnm = cnm.fit(images)
    return cnm

def save_cnmf(cnm):
    """
    Saves the CNMF object to a file.
    Args:
        cnm (CNMF object): Fitted CNMF object.
    """
    cnm.save('test_file.hdf5')

def load_saved_cnmf():
    """
    Loads the saved CNMF object from a file.
    Returns:
        cnm2 (CNMF object): Loaded CNMF object.
    """
    cnm2 = cnmf.cnmf.load_CNMF('test_file.hdf5')
    return cnm2

def test_cnmf():
    """
    Runs the CNMF demo and performs assertions.
    """
    Yr, dims, T = load_movie()
    cnm = initialize_cnmf(2, None)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    cnm = fit_cnmf(cnm, images)
    save_cnmf(cnm)
    cnm2 = load_saved_cnmf()
    npt.assert_allclose(cnm.estimates.A.sum(), cnm2.estimates.A.sum())
    npt.assert_allclose(cnm.estimates.C, cnm2.estimates.C)

test_cnmf()