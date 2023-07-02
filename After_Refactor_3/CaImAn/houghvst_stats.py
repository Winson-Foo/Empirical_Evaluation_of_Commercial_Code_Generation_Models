import numpy as np
from scipy import stats


def add_poisson_gaussian_noise(arr, sigma, alpha, mu=0):
    """
    Add Poisson-Gaussian noise to an array.

    Parameters:
        arr (ndarray): Input array.
        sigma (float): Standard deviation of Gaussian noise.
        alpha (float): Scaling factor for Poisson noise.
        mu (float): Mean of Gaussian noise.

    Returns:
        ndarray: Noisy array.
    """
    poisson_noise = stats.poisson.rvs(arr)
    gaussian_noise = stats.norm.rvs(scale=sigma, loc=mu, size=arr.shape)
    return alpha * poisson_noise + gaussian_noise


def add_gaussian_noise(arr, sigma):
    """
    Add Gaussian noise to an array.

    Parameters:
        arr (ndarray): Input array.
        sigma (float): Standard deviation of Gaussian noise.

    Returns:
        ndarray: Noisy array.
    """
    noise = stats.norm.rvs(scale=sigma, size=arr.shape)
    return arr + noise


def calculate_std_mad(a, loc=None, axis=None):
    """
    Calculate the standard median absolute deviation of an array.

    Parameters:
        a (ndarray): Input array.
        loc (float or ndarray, optional): Location parameter for defining deviations.
        axis (int or None, optional): Axis or axes along which to compute the median absolute deviation.

    Returns:
        ndarray: Standard median absolute deviation.
    """
    if loc is None:
        loc = np.median(a, axis=axis, keepdims=True)
    return np.median(np.abs(a - loc), axis=axis) / 0.6744897501960817


def calculate_std_sn(a):
    """
    Calculate the standard skedastic noise of an array.

    Parameters:
        a (ndarray): Input array.

    Returns:
        float: Standard skedastic noise.
    """
    diff = np.abs(a[:, np.newaxis] - a[np.newaxis, :])
    return 1.1926 * np.median(np.median(diff, axis=0))


def calculate_half_sample_mode(x, sort=True, axis=None):
    """
    Calculate the half-sample mode of an array.

    Parameters:
        x (ndarray): Input array.
        sort (bool, optional): Whether to sort the array before calculation.
        axis (int or None, optional): Axis along which to compute the mode.

    Returns:
        ndarray: Half-sample mode of the array.
    """
    if axis is not None:
        return np.apply_along_axis(calculate_half_sample_mode, axis, x)

    if len(x) <= 2:
        return np.mean(x, axis=axis)
    if len(x) == 3:
        if np.mean(x[1:]) < np.mean(x[:-1]):
            return np.mean(x[1:], axis=axis)
        if np.mean(x[1:]) > np.mean(x[:-1]):
            return np.mean(x[:-1], axis=axis)
        return x[1]

    if sort:
        sorted_x = np.sort(x, axis=axis)
    else:
        sorted_x = x
    half_idx = (len(x) + 1) // 2

    ranges = sorted_x[-half_idx:] - sorted_x[:half_idx]
    smallest_range_idx = np.argmin(ranges)

    x_subset = sorted_x[smallest_range_idx:(smallest_range_idx + half_idx)]
    return calculate_half_sample_mode(x_subset, sort=False)


def calculate_half_range_mode(x, sort=True):
    """
    Calculate the half-range mode of an array.

    Parameters:
        x (ndarray): Input array.
        sort (bool, optional): Whether to sort the array before calculation.

    Returns:
        ndarray: Half-range mode of the array.
    """
    if len(x) <= 2:
        return np.mean(x)
    if len(x) == 3:
        if np.mean(x[1:]) < np.mean(x[:-1]):
            return np.mean(x[1:])
        if np.mean(x[1:]) > np.mean(x[:-1]):
            return np.mean(x[:-1])
        return x[1]

    if sort:
        sorted_x = np.sort(x)
    else:
        sorted_x = x
    half_idx = (len(x) + 1) // 2

    ranges = sorted_x[-half_idx:] - sorted_x[:half_idx]
    smallest_range_idx = np.argmin(ranges)

    x_subset = sorted_x[smallest_range_idx:(smallest_range_idx + half_idx)]
    return calculate_half_sample_mode(x_subset, sort=False)