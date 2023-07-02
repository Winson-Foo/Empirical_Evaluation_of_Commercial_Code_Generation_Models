def anscombe_transform(arr, alpha=1):
    """
    Anscombe variance-stabilizing transformation
    """
    return np.sqrt(arr + 3/8) * alpha

def inverse_anscombe_transform(arr, alpha=1):
    """
    Inverse of the Anscombe variance-stabilizing transformation
    """
    return ((arr / alpha)**2 - 3/8) / alpha

def generalized_anscombe_transform(arr, sigma_sq, alpha=1):
    """
    Generalized Anscombe variance-stabilizing transformation
    """
    v = np.maximum((arr / alpha) + (3. / 8.) + sigma_sq / (alpha**2), 0)
    f = 2. * np.sqrt(v)
    return f

def inverse_generalized_anscombe_transform(arr, sigma_sq, m=0, alpha=1, method='asym'):
    """
    Inverse of the Generalized Anscombe variance-stabilizing transformation
    """
    sigma_sq /= alpha**2

    if method == 'closed-form':
        # closed-form approximation of the exact unbiased inverse:
        arr_trunc = np.maximum(arr, 0.8)
        inverse = ((arr_trunc / 2.)**2 + 0.25 * np.sqrt(1.5) * arr_trunc**-1 - (11. / 8.) * arr_trunc**-2 +
                   (5. / 8.) * np.sqrt(1.5) * arr_trunc**-3 - (1. / 8.) - sigma_sq)
    elif method == 'asym':
        # asymptotic approximation of the exact unbiased inverse:
        inverse = (arr / 2.)**2 - 1. / 8 - sigma_sq
        # inverse = np.maximum(0, inverse)
    else:
        raise NotImplementedError('Only supports the closed-form')

    if alpha != 1:
        inverse *= alpha

    if m != 0:
        inverse += m

    return inverse