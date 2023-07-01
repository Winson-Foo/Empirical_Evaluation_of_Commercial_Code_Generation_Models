import eagerpy as ep

def l2_clipping_aware_rescaling(x, delta, eps: float, a: float = 0.0, b: float = 1.0):
    """Calculates eta such that norm(clip(x + eta * delta, a, b) - x) == eps.

    Assumes x and delta have a batch dimension and eps, a, b, and p are
    scalars. If the equation cannot be solved because eps is too large, the
    left hand side is maximized.

    Args:
        x: A batch of inputs.
        delta: A batch of perturbation directions (same shape as x).
        eps: The target norm (non-negative float).
        a: The lower bound of the data domain (float).
        b: The upper bound of the data domain (float).

    Returns:
        eta: A batch of scales with the same number of dimensions as x but all
            axis == 1 except for the batch dimension.
    """
    (x, delta), restore_fn = ep.astensors_(x, delta)
    num_samples = x.shape[0]
    assert delta.shape[0] == num_samples
    sample_indices = ep.arange(x, num_samples)

    delta_squared = delta.square().reshape((num_samples, -1))
    space = ep.where(delta >= 0, b - x, x - a).reshape((num_samples, -1))
    f_squared = space.square() / ep.maximum(delta_squared, 1e-20)
    sorted_indices = ep.argsort(f_squared, axis=-1)
    sorted_f_squared = f_squared[sample_indices[:, ep.newaxis], sorted_indices]
    m = ep.cumsum(delta_squared[sample_indices[:, ep.newaxis], sorted_indices.flip(axis=1)], axis=-1).flip(axis=1)
    dx = sorted_f_squared[:, 1:] - sorted_f_squared[:, :-1]
    dx = ep.concatenate((sorted_f_squared[:, :1], dx), axis=-1)
    dy = m * dx
    y = ep.cumsum(dy, axis=-1)
    c = y >= eps**2

    f = ep.arange(x, c.shape[-1], 0, -1)
    j = ep.argmax(c.astype(f.dtype) * f, axis=-1)

    eta_squared = sorted_f_squared[sample_indices, j] - (y[sample_indices, j] - eps**2) / m[sample_indices, j]
    eta_squared = ep.where(c[:, -1], eta_squared, sorted_f_squared[:, -1])
    eta = ep.sqrt(eta_squared)
    eta = eta.reshape((-1,) + (1,) * (x.ndim - 1))

    return restore_fn(eta)