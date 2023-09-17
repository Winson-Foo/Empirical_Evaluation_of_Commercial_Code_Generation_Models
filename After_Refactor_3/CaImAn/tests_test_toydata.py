#!/usr/bin/env python

import numpy as np
import numpy.testing as npt
from scipy.ndimage.filters import gaussian_filter

import caiman.source_extraction.cnmf.params
from caiman.source_extraction import cnmf as cnmf


def generate_data(D: int = 3, noise: float = 0.5, T: int = 300, framerate: int = 30, firerate: float = 2.):
    N = 4  # number of neurons
    dims = [(20, 30), (12, 14, 16)][D - 2]  # size of image
    sig = (2, 2, 2)[:D]  # neurons size
    bkgrd = 10  # fluorescence baseline
    gamma = .9  # calcium decay time constant

    np.random.seed(5)
    centers = np.asarray([[np.random.randint(4, x - 4) for x in dims] for _ in range(N)])

    trueA = np.zeros(dims + (N,), dtype=np.float32)
    trueS = np.random.rand(N, T) < firerate / float(framerate)
    trueS[:, 0] = 0
    trueC = trueS.astype(np.float32)

    for i in range(1, T):
        trueC[:, i] += gamma * trueC[:, i - 1]

    for i in range(N):
        trueA[tuple(centers[i]) + (i,)] = 1.

    tmp = np.zeros(dims)
    tmp[tuple(d // 2 for d in dims)] = 1.
    z = np.linalg.norm(gaussian_filter(tmp, sig).ravel())
    trueA = 10 * gaussian_filter(trueA, sig + (0,)) / z

    Yr = bkgrd + noise * np.random.randn(*(np.prod(dims), T)) + trueA.reshape((-1, 4), order='F').dot(trueC)

    return Yr, trueC, trueS, trueA, centers, dims


def fit_cnmf(Yr: np.ndarray, dims: tuple, params: caiman.source_extraction.cnmf.params.CNMFParams):
    cnm = cnmf.CNMF(2, params=params)
    images = np.reshape(Yr.T, (Yr.shape[1],) + dims, order='F')
    cnm = cnm.fit(images)
    return cnm


def verify_correlation(cnm, trueC, trueA):
    N = trueC.shape[0]
    sorting = [np.argmax([np.corrcoef(tc, c)[0, 1] for tc in trueC]) for c in cnm.estimates.C]

    # verifying the temporal components
    corr_temp = [np.corrcoef(trueC[sorting[i]], cnm.estimates.C[i])[0, 1] for i in range(N)]
    npt.assert_allclose(corr_temp, 1, .05)

    # verifying the spatial components
    corr_spatial = [
        np.corrcoef(np.reshape(trueA, (-1, 4), order='F')[:, sorting[i]],
                    cnm.estimates.A.toarray()[:, i])[0, 1] for i in range(N)
    ]
    npt.assert_allclose(corr_spatial, 1, .05)


def pipeline(D: int):
    Yr, trueC, trueS, trueA, centers, dims = generate_data(D)
    N, T = trueC.shape

    params = caiman.source_extraction.cnmf.params.CNMFParams(dims=dims,
                                                             k=4,
                                                             gSig=[2, 2, 2][:D],
                                                             p=1,
                                                             n_pixels_per_process=np.prod(dims),
                                                             block_size_spat=np.prod(dims),
                                                             block_size_temp=np.prod(dims))
    params.spatial['thr_method'] = 'nrg'
    params.spatial['extract_cc'] = False

    cnm = fit_cnmf(Yr, dims, params)

    verify_correlation(cnm, trueC, trueA)


def test_2D():
    pipeline(2)


def test_3D():
    pipeline(3)