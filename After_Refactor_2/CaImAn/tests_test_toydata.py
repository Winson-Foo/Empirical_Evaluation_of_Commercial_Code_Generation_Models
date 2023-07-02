import numpy.testing as npt
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.params import CNMFParams


def gen_data(D: int = 3, noise: float = .5, T: int = 300, framerate: int = 30, firerate: float = 2.) -> tuple:
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
    Yr = bkgrd + noise * np.random.randn(*(np.prod(dims), T)) + \
        trueA.reshape((-1, 4), order='F').dot(trueC)
    return Yr, trueC, trueS, trueA, centers, dims


def fit_cnmf(images: np.ndarray, params: CNMFParams) -> cnmf.CNMF:
    cnm = cnmf.CNMF(2, params=params)
    cnm = cnm.fit(images)
    return cnm


def verify_correlation(trueC: np.ndarray, cnmC: np.ndarray) -> None:
    sorting = [np.argmax([np.corrcoef(tc, c)[0, 1] for tc in trueC]) for c in cnmC]
    corr = [np.corrcoef(trueC[sorting[i]], cnmC[i])[0, 1] for i in range(len(trueC))]
    npt.assert_allclose(corr, 1, .05)


def verify_spatial_components(trueA: np.ndarray, cnmA: np.ndarray) -> None:
    sorting = [np.argmax([np.corrcoef(trueA[:, sorting[i]], a)[0, 1] for a in cnmA.T]) for i in range(len(trueA[0]))]
    corr = [np.corrcoef(np.reshape(trueA, (-1, len(trueA[0]))), cnmA.T[:, sorting[i]])[0, 1] for i in range(len(trueA[0]))]
    npt.assert_allclose(corr, 1, .05)


def pipeline(D: int) -> None:
    # GENERATE GROUND TRUTH DATA
    Yr, trueC, trueS, trueA, centers, dims = gen_data(D)
    N, T = trueC.shape

    # INIT
    params = CNMFParams(dims=dims,
                        k=4,
                        gSig=[2, 2, 2][:D],
                        p=1,
                        n_pixels_per_process=np.prod(dims),
                        block_size_spat=np.prod(dims),
                        block_size_temp=np.prod(dims))
    params.spatial['thr_method'] = 'nrg'
    params.spatial['extract_cc'] = False

    # FIT
    images = np.reshape(Yr.T, (T,) + dims, order='F')
    cnm = fit_cnmf(images, params)

    # VERIFY HIGH CORRELATION WITH GROUND TRUTH
    verify_correlation(trueC, cnm.estimates.C)
    verify_spatial_components(trueA, cnm.estimates.A.toarray())


def test_2D():
    pipeline(2)


def test_3D():
    pipeline(3)