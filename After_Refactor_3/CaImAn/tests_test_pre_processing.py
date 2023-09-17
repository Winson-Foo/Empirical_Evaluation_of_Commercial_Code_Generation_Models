#!/usr/bin/env python

import numpy as np
from caiman.source_extraction import cnmf as cnmf


def calculate_autocovariance(data: np.ndarray, maxlag: int) -> np.ndarray:
    autocov = cnmf.pre_processing.axcov(data, maxlag)
    return autocov


def test_axcov() -> None:
    data = np.random.randn(1000)
    maxlag = 5
    autocov = calculate_autocovariance(data, maxlag)
    print(autocov)

    expected_result = np.concatenate((np.zeros(maxlag), np.array([1]), np.zeros(maxlag)))
    np.testing.assert_allclose(autocov, expected_result, atol=1)


if __name__ == "__main__":
    test_axcov()