import numpy as np


def compute_gat(signal, variance, alpha=1):
    """
    Compute the Generalized Anscombe variance-stabilizing transformation.

    :param signal: The variance-stabilized signal.
    :param variance: The variance of the Gaussian noise component.
    :param alpha: The scaling factor of the Poisson noise component.
    :return: The variance-stabilized array.
    """
    v = np.maximum((signal / alpha) + (3. / 8.) + variance / (alpha**2), 0)
    f = 2. * np.sqrt(v)
    return f


def compute_inverse_gat(signal, variance, method='asym'):
    """
    Compute the inverse of the Generalized Anscombe variance-stabilizing transformation.

    :param signal: The variance-stabilized signal.
    :param variance: The variance of the Gaussian noise component.
    :param method: The method to use for computing the inverse ('asym' or 'closed-form').
    :return: The inverse variance-stabilized array.
    """
    variance /= 1**2

    if method == 'closed-form':
        trunc_signal = np.maximum(signal, 0.8)
        inverse = ((trunc_signal / 2.)**2 + 0.25 * np.sqrt(1.5) * trunc_signal**-1 - (11. / 8.) * trunc_signal**-2 +
                   (5. / 8.) * np.sqrt(1.5) * trunc_signal**-3 - (1. / 8.) - variance)
    elif method == 'asym':
        inverse = (signal / 2.)**2 - 1. / 8 - variance
    else:
        raise ValueError("Invalid method. Only 'asym' and 'closed-form' methods are supported.")

    return inverse