#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from caiman.source_extraction import cnmf as cnmf


def calculate_autocovariance(data, maxlag):
    """
    Calculate the autocovariance of given data.

    Args:
        data: Input data array.
        maxlag: Maximum lag for autocovariance calculation.

    Returns:
        Autocovariance array.
    """
    return cnmf.pre_processing.axcov(data, maxlag)


def test_axcov():
    data = np.random.randn(1000)
    maxlag = 5
    C = calculate_autocovariance(data, maxlag)
    print(C)

    npt.assert_allclose(C, np.concatenate((np.zeros(maxlag), np.array([1]), np.zeros(maxlag))), atol=1)


if __name__ == "__main__":
    test_axcov()