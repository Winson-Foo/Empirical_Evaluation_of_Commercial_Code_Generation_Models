from scipy import stats


def poisson_gaussian_noise(arr, sigma, alpha, mu=0):
    p = stats.poisson.rvs(arr)
    n = stats.norm.rvs(scale=sigma, loc=mu, size=arr.shape)
    return alpha * p + n


def gaussian_noise(arr, sigma):
    n = stats.norm.rvs(scale=sigma, size=arr.shape)
    return arr + n


def median_absolute_deviation(array, loc=None, axis=None):
    if loc is None:
        loc = np.median(array, axis=axis, keepdims=True)
    return np.median(np.abs(array - loc), axis=axis) / 0.6744897501960817


def std_sn(data):
    diff = np.abs(data[:, np.newaxis] - data[np.newaxis, :])
    return 1.1926 * np.median(np.median(diff, axis=0))


def half_sample_mode(data, sort=True, axis=None):
    """
    Estimate the mode of the half sample of the data. 
    Based on the algorithm from [1] with modifications to handle len(data) == 3.
    
    Parameters:
    - data: The input data.
    - sort: Whether to sort the data before processing.
    - axis: The axis along which to calculate the mode.
    
    Returns:
    - The estimated mode of the half sample.
    
    [1] http://arxiv.org/abs/math.ST/0505419
    """
    if axis is not None:
        return np.apply_along_axis(half_sample_mode, axis, data)

    if len(data) <= 2:
        return np.mean(data, axis=axis)
    if len(data) == 3:
        if np.mean(data[1:]) < np.mean(data[:-1]):
            return np.mean(data[1:], axis=axis)
        if np.mean(data[1:]) > np.mean(data[:-1]):
            return np.mean(data[:-1], axis=axis)
        return data[1]

    if sort:
        sorted_data = np.sort(data, axis=axis)
    else:
        sorted_data = data

    half_idx = (len(data) + 1) // 2
    ranges = sorted_data[-half_idx:] - sorted_data[:half_idx]
    smallest_range_idx = np.argmin(ranges)

    data_subset = sorted_data[smallest_range_idx:(smallest_range_idx + half_idx)]
    return half_sample_mode(data_subset, sort=False)


def half_range_mode(data, sort=True):
    """
    Estimate the mode of the half range of the data.
    Based on the algorithm from [1] with modifications to handle len(data) == 3.
    
    Parameters:
    - data: The input data.
    - sort: Whether to sort the data before processing.
    
    Returns:
    - The estimated mode of the half range.
    
    [1] http://arxiv.org/abs/math.ST/0505419
    """
    if len(data) <= 2:
        return np.mean(data)
    if len(data) == 3:
        if np.mean(data[1:]) < np.mean(data[:-1]):
            return np.mean(data[1:])
        if np.mean(data[1:]) > np.mean(data[:-1]):
            return np.mean(data[:-1])
        return data[1]

    if sort:
        sorted_data = np.sort(data)
    else:
        sorted_data = data

    half_idx = (len(data) + 1) // 2
    ranges = sorted_data[-half_idx:] - sorted_data[:half_idx]
    smallest_range_idx = np.argmin(ranges)

    data_subset = sorted_data[smallest_range_idx:(smallest_range_idx + half_idx)]
    return half_sample_mode(data_subset, sort=False)