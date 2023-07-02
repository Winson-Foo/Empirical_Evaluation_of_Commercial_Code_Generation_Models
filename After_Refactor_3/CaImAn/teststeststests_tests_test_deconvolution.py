#!/usr/bin/env python

import logging
from time import time

import numpy as np
import numpy.testing as npt

from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi


def setup_logging():
    """
    Set up the logger.
    Change this function if you want to log to a file using the filename parameter,
    or make the output more or less verbose by setting the level.
    """
    logging.basicConfig(
        format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s",
        level=logging.DEBUG
    )


def generate_data(g=[.95], sn=.2, T=1000, framerate=30, firerate=.5, b=10, N=1, seed=0):
    """
    Generate data from homogenous Poisson Process

    Parameters
    ----------
    g : list, optional, default=[.95]
        Parameter(s) of the AR(p) process that models the fluorescence impulse response.
    sn : float, optional, default .2
        Noise standard deviation.
    T : int, optional, default 1000
        Duration.
    framerate : int, optional, default 30
        Frame rate.
    firerate : int, optional, default .5
        Neural firing rate.
    b : int, optional, default 10
        Baseline.
    N : int, optional, default 1
        Number of generated traces.
    seed : int, optional, default 0
        Seed of random number generator.

    Returns
    -------
    Y : array, shape (N, T)
        Noisy fluorescence data.
    trueC : array, shape (N, T)
        Calcium traces (without sn).
    trueS : array, shape (N, T)
        Spike trains.
    """
    np.random.seed(seed)
    Y = np.zeros((N, T))
    trueS = np.random.rand(N, T) < firerate / float(framerate)
    trueC = trueS.astype(float)

    for i in range(2, T):
        if len(g) == 2:
            trueC[:, i] += g[0] * trueC[:, i - 1] + g[1] * trueC[:, i - 2]
        else:
            trueC[:, i] += g[0] * trueC[:, i - 1]
    Y = b + trueC + sn * np.random.randn(N, T)

    return Y, trueC, trueS


def run_foopsi_test(method: str, p: int):
    """
    Run constrained_foopsi test for a given method and p.

    Parameters
    ----------
    method : str
        Foopsi method ('oasis' in this case).
    p : int
        AR model order.
    """
    t = time()
    g = np.array([[.95], [1.7, -.71]][p - 1])

    for i, sn in enumerate([.2, .5]):  # high and low SNR
        y, c, s = generate_data(g, sn)
        res = constrained_foopsi(y, g=g, sn=sn, p=p, method=method)
        npt.assert_allclose(np.corrcoef(res[0], c)[0, 1], 1, [.01, .1][i])
        npt.assert_allclose(np.corrcoef(res[-2], s)[0, 1], 1, [.03, .3][i])

    logging.debug(['\n', ''][p - 1] + ' %5s AR%d   %.4fs' % (method, p, time() - t))


def test_oasis():
    """
    Run the oasis test for different model orders.
    """
    run_foopsi_test('oasis', 1)
    run_foopsi_test('oasis', 2)


if __name__ == "__main__":
    setup_logging()
    test_oasis()