#!/usr/bin/env python

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from caiman.source_extraction.cnmf import utilities


def test_agreement_gaussian_filter(D):
    """Test agreement with scipy.ndimage gaussian_filter"""
    img = np.random.rand(*(10,) * D)
    gSig = (5, 3, 2, 4)[:D]
    for mode in ('nearest', 'reflect', 'constant', 'mirror'):
        for truncate in (2, 4):
            scipy_result = ndi.gaussian_filter(img, gSig, mode=mode, truncate=truncate, cval=2)
            caiman_result = utilities.gaussian_filter(img, gSig, mode=mode, truncate=truncate, cval=2)
            np.testing.assert_allclose(scipy_result, caiman_result)


def test_agreement_uniform_filter(D):
    """Test agreement with scipy.ndimage uniform_filter"""
    img = np.random.rand(*(10,) * D)
    size = (5, 3, 2, 4)[:D]
    for mode in ('nearest', 'reflect', 'constant', 'mirror'):
        scipy_result = ndi.uniform_filter(img, size, mode=mode, cval=2)
        caiman_result = utilities.uniform_filter(img, size, mode=mode, cval=2)
        np.testing.assert_allclose(scipy_result, caiman_result)


def test_agreement_maximum_filter(D):
    """Test agreement with scipy.ndimage maximum_filter"""
    img = np.random.rand(*(10,) * D)
    size = (5, 3, 2, 4)[:D]
    for mode in ('nearest', 'reflect', 'constant', 'mirror'):
        scipy_result = ndi.maximum_filter(img, size, mode=mode)
        caiman_result = utilities.maximum_filter(img, size, mode=mode)
        np.testing.assert_allclose(scipy_result, caiman_result)


def test_peak_local_max():
    """Test peak_local_max function"""
    for D in (1, 2, 3):
        img = np.random.rand(*(10,) * D)
        for m in (1, 2, 3, 4):
            np.testing.assert_array_equal(peak_local_max(img, min_distance=m),
                                          utilities.peak_local_max(img, min_distance=m))


def test_filter_output(D):
    """Test array in which to place the filter output"""
    img = np.random.rand(*(10,) * D)
    out = np.zeros_like(img)
    tmp = utilities.gaussian_filter(img, (5, 3, 2, 4)[:D], output=out)
    np.testing.assert_array_equal(out, tmp)
    tmp = utilities.gaussian_filter(img, (5, 3, 2, 4)[:D], output=out, mode='constant', cval=2)
    np.testing.assert_array_equal(out, tmp)


def test_gaussian_filter():
    """Test gaussian_filter functions"""
    test_agreement_gaussian_filter(1)
    test_agreement_gaussian_filter(2)
    test_agreement_gaussian_filter(3)
    test_agreement_gaussian_filter(4)
    test_filter_output(1)
    test_filter_output(2)
    test_filter_output(3)


def test_uniform_filter():
    """Test uniform_filter functions"""
    test_agreement_uniform_filter(1)
    test_agreement_uniform_filter(2)
    test_agreement_uniform_filter(3)
    test_filter_output(1)
    test_filter_output(2)
    test_filter_output(3)


def test_maximum_filter():
    """Test maximum_filter functions"""
    test_agreement_maximum_filter(1)
    test_agreement_maximum_filter(2)
    test_agreement_maximum_filter(3)
    test_filter_output(1)
    test_filter_output(2)
    test_filter_output(3)