import numpy.testing as npt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from caiman.source_extraction.cnmf import utilities


# Gaussian Filter Functions

def test_gaussian_filter_1d():
    """
    Test the agreement of the Gaussian filter with scipy.ndimage for 1D images.
    """
    _test_gaussian_filter(1)


def test_gaussian_filter():
    """
    Test the agreement of the Gaussian filter with scipy.ndimage for 2D images.
    """
    _test_gaussian_filter(2)


def test_gaussian_filter_3d():
    """
    Test the agreement of the Gaussian filter with scipy.ndimage for 3D images.
    """
    _test_gaussian_filter(3)


def test_gaussian_filter_4d():
    """
    Test the agreement of the Gaussian filter with scipy.ndimage for 4D images.
    """
    _test_gaussian_filter(4)


def test_gaussian_filter_output():
    """
    Test the output of the Gaussian filter function.
    """
    for dimensions in (1, 2, 3):
        image = np.random.rand(*([10] * dimensions))
        output = np.zeros_like(image)
        result = utilities.gaussian_filter(image, (5, 3, 2, 4)[:dimensions], output=output)
        npt.assert_array_equal(output, result)
        result = utilities.gaussian_filter(image, (5, 3, 2, 4)[:dimensions], output=output, mode='constant', cval=2)
        npt.assert_array_equal(output, result)


# Uniform Filter Functions

def test_uniform_filter_1d():
    """
    Test the agreement of the uniform filter with scipy.ndimage for 1D images.
    """
    _test_uniform_filter(1)


def test_uniform_filter():
    """
    Test the agreement of the uniform filter with scipy.ndimage for 2D images.
    """
    _test_uniform_filter(2)


def test_uniform_filter_3d():
    """
    Test the agreement of the uniform filter with scipy.ndimage for 3D images.
    """
    _test_uniform_filter(3)


def test_uniform_filter_output():
    """
    Test the output of the uniform filter function.
    """
    for dimensions in (1, 2, 3):
        image = np.random.rand(*([10] * dimensions))
        output = np.zeros_like(image)
        result = utilities.uniform_filter(image, (5, 3, 2, 4)[:dimensions], output=output)
        npt.assert_array_equal(output, result)
        result = utilities.uniform_filter(image, (5, 3, 2, 4)[:dimensions], output=output, mode='constant', cval=2)
        npt.assert_array_equal(output, result)


# Maximum Filter Functions

def test_maximum_filter_1d():
    """
    Test the agreement of the maximum filter with scipy.ndimage for 1D images.
    """
    _test_maximum_filter(1)


def test_maximum_filter():
    """
    Test the agreement of the maximum filter with scipy.ndimage for 2D images.
    """
    _test_maximum_filter(2)


def test_maximum_filter_3d():
    """
    Test the agreement of the maximum filter with scipy.ndimage for 3D images.
    """
    _test_maximum_filter(3)


def test_maximum_filter_output():
    """
    Test the output of the maximum filter function.
    """
    for dimensions in (1, 2, 3):
        image = np.random.rand(*([10] * dimensions))
        output = np.zeros_like(image)
        result = utilities.maximum_filter(image, (5, 3, 2, 4)[:dimensions], output=output)
        npt.assert_array_equal(output, result)


# Peak Local Max Function

def test_peak_local_max():
    """
    Test the peak_local_max function.
    """
    for dimensions in (1, 2, 3):
        image = np.random.rand(*([10] * dimensions))
        for min_distance in (1, 2, 3, 4):
            npt.assert_array_equal(
                peak_local_max(image, min_distance=min_distance),
                utilities.peak_local_max(image, min_distance=min_distance)
            )