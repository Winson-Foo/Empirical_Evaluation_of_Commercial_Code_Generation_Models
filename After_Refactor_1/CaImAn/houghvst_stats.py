def poisson_gaussian_noise(arr, sigma, alpha, mu=0):
    """
    Add Poisson and Gaussian noise to the input array.
    """
    poisson_noise = add_poisson_noise(arr)
    gaussian_noise = add_gaussian_noise(arr, sigma, mu)
    return alpha * poisson_noise + gaussian_noise


def add_poisson_noise(arr):
    """
    Add Poisson noise to the input array.
    """
    return np.random.poisson(arr)


def add_gaussian_noise(arr, sigma, mu=0):
    """
    Add Gaussian noise to the input array.
    """
    return arr + np.random.normal(loc=mu, scale=sigma, size=arr.shape)


def std_mad(a, loc=None, axis=None):
    """
    Calculate the median absolute deviation (MAD) of the input array.
    """
    if loc is None:
        loc = np.median(a, axis=axis, keepdims=True)
    return np.median(np.abs(a - loc), axis=axis) / 0.6744897501960817


def std_sn(a):
    """
    Calculate the standard Sn of the input array.
    """
    diff = np.abs(a[:, np.newaxis] - a[np.newaxis, :])
    return 1.1926 * np.median(np.median(diff, axis=0))


def half_sample_mode(x, sort=True, axis=None):
    """
    Calculate the half-sample mode estimate of the input array.
    """
    if axis is not None:
        return np.apply_along_axis(half_sample_mode, axis, x)

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
    return half_sample_mode(x_subset, sort=False)


def half_range_mode(x, sort=True):
    """
    Calculate the half-range mode estimate of the input array.
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
    return half_sample_mode(x_subset, sort=False)