#!/usr/bin/env python

import numpy as np
from scipy.ndimage.filters import gaussian_filter
import numpy.testing as npt

from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.params import CNMFParams


def generate_data(dimension=3, noise=0.5, frames=300, framerate=30, firerate=2.0):
    n_neurons = 4  # number of neurons
    dimensions = [(20, 30), (12, 14, 16)][dimension - 2]  # size of image
    sig = (2, 2, 2)[:dimension]  # neurons size
    bkgrd = 10  # fluorescence baseline
    gamma = 0.9  # calcium decay time constant

    np.random.seed(5)
    centers = np.asarray([[np.random.randint(4, x - 4) for x in dimensions] for _ in range(n_neurons)])
    trueA = np.zeros(dimensions + (n_neurons,), dtype=np.float32)
    trueS = np.random.rand(n_neurons, frames) < firerate / float(framerate)
    trueS[:, 0] = 0
    trueC = trueS.astype(np.float32)
    
    for i in range(1, frames):
        trueC[:, i] += gamma * trueC[:, i - 1]
    
    for i in range(n_neurons):
        trueA[tuple(centers[i]) + (i,)] = 1.
    
    tmp = np.zeros(dimensions)
    tmp[tuple(d // 2 for d in dimensions)] = 1.
    z = np.linalg.norm(gaussian_filter(tmp, sig).ravel())
    trueA = 10 * gaussian_filter(trueA, sig + (0,)) / z
    Yr = bkgrd + noise * np.random.randn(*(np.prod(dimensions), frames)) + \
        trueA.reshape((-1, 4), order='F').dot(trueC)
    
    return Yr, trueC, trueS, trueA, centers, dimensions


def run_pipeline(dimension):
    Yr, trueC, trueS, trueA, centers, dimensions = generate_data(dimension)
    n_neurons, frames = trueC.shape
    
    params = CNMFParams(dims=dimensions,
                        k=4,
                        gSig=[2, 2, 2][:dimension],
                        p=1,
                        n_pixels_per_process=np.prod(dimensions),
                        block_size_spat=np.prod(dimensions),
                        block_size_temp=np.prod(dimensions))
    
    params.spatial['thr_method'] = 'nrg'
    params.spatial['extract_cc'] = False
    
    cnm = cnmf.CNMF(2, params=params)
    
    images = np.reshape(Yr.T, (frames,) + dimensions, order='F')
    cnm = cnm.fit(images)
    
    sorting = [np.argmax([np.corrcoef(tc, c)[0, 1] for tc in trueC]) for c in cnm.estimates.C]
    
    corr = [np.corrcoef(trueC[sorting[i]], cnm.estimates.C[i])[0, 1] for i in range(n_neurons)]
    npt.assert_allclose(corr, 1, 0.05)
    
    corr = [
        np.corrcoef(np.reshape(trueA, (-1, 4), order='F')[:, sorting[i]],
                    cnm.estimates.A.toarray()[:, i])[0, 1] for i in range(n_neurons)
    ]
    npt.assert_allclose(corr, 1, 0.05)


def test_2D_pipeline():
    run_pipeline(2)


def test_3D_pipeline():
    run_pipeline(3)