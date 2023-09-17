#!/usr/bin/env python

import numpy.testing as npt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from caiman.source_extraction.cnmf import utilities


def test_gaussian_filter():
    """
    Test agreement with scipy.ndimage for gaussian_filter function.
    """
    sizes = [(5, 3, 2, 4)[:D] for D in (1, 2, 3, 4)]
    modes = ['nearest', 'reflect', 'constant', 'mirror']
    truncates = [2, 4]

    for size in sizes:
        for mode in modes:
            for truncate in truncates:
                img = np.random.rand(*(10,) * len(size))
                sc = ndi.gaussian_filter(img, size, mode=mode, truncate=truncate, cval=2)
                cv = utilities.gaussian_filter(img, size, mode=mode, truncate=truncate, cval=2)
                npt.assert_allclose(sc, cv)


def test_uniform_filter():
    """
    Test agreement with scipy.ndimage for uniform_filter function.
    """
    sizes = [(5, 3, 2, 4)[:D] for D in (1, 2, 3)]
    modes = ['nearest', 'reflect', 'constant', 'mirror']

    for size in sizes:
        for mode in modes:
            img = np.random.rand(*(10,) * len(size))
            sc = ndi.uniform_filter(img, size, mode=mode, cval=2)
            cv = utilities.uniform_filter(img, size, mode=mode, cval=2)
            npt.assert_allclose(sc, cv)


def test_maximum_filter():
    """
    Test agreement with scipy.ndimage for maximum_filter function.
    """
    sizes = [(5, 3, 2, 4)[:D] for D in (1, 2, 3)]
    modes = ['nearest', 'reflect', 'constant', 'mirror']

    for size in sizes:
        for mode in modes:
            img = np.random.rand(*(10,) * len(size))
            sc = ndi.maximum_filter(img, size, mode=mode)
            cv = utilities.maximum_filter(img, size, mode=mode)
            npt.assert_allclose(sc, cv)


def test_filter_output():
    """
    Test output correctness for gaussian_filter, uniform_filter, and maximum_filter functions.
    """
    sizes = [(5, 3, 2, 4)[:D] for D in (1, 2, 3)]
    modes = ['nearest', 'reflect', 'constant', 'mirror']

    for size in sizes:
        for mode in modes:
            img = np.random.rand(*(10,) * len(size))
            out = np.zeros_like(img)
            
            cv = utilities.gaussian_filter(img, size, output=out, mode=mode, cval=2)
            npt.assert_array_equal(out, cv)
            
            cv = utilities.uniform_filter(img, size, output=out, mode=mode, cval=2)
            npt.assert_array_equal(out, cv)
            
            cv = utilities.maximum_filter(img, size, output=out, mode=mode)
            npt.assert_array_equal(out, cv)


def test_peak_local_max():
    """
    Test agreement with skimage.feature.peak_local_max function.
    """
    distances = [1, 2, 3, 4]
    sizes = [(10,) * D for D in (1, 2, 3)]

    for size in sizes:
        img = np.random.rand(*size)
        
        for distance in distances:
            sc = peak_local_max(img, min_distance=distance)
            cv = utilities.peak_local_max(img, min_distance=distance)
            npt.assert_array_equal(sc, cv)