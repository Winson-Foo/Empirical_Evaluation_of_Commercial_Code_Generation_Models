#!/usr/bin/env python

import numpy as np
import numpy.testing as npt
from caiman.source_extraction import cnmf

def calculate_C(data, maxlag):
    """
    Calculate C using axcov function from cnmf.pre_processing module.
    
    Args:
        data: Input data as a 1D numpy array.
        maxlag: Maximum lag for autocovariance computation.
    
    Returns:
        C: Autocovariance computed using axcov function.
    """
    C = cnmf.pre_processing.axcov(data, maxlag)
    return C

def test_axcov():
    """
    Test function for axcov.
    """
    data = np.random.randn(1000)
    maxlag = 5
    C = calculate_C(data, maxlag)
    npt.assert_allclose(C, np.concatenate((np.zeros(maxlag), np.array([1]), np.zeros(maxlag))), atol=1)

if __name__ == '__main__':
    test_axcov()