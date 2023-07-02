import eagerpy as ep

def l2_clipping_aware_rescaling(x, delta, eps: float, a: float = 0.0, b: float = 1.0) -> ep.Tensor:
    """Calculates eta such that norm(clip(x + eta * delta, a, b) - x) == eps.

    Assumes x and delta have a batch dimension and eps, a, b, and p are
    scalars. If the equation cannot be solved because eps is too large, the
    left hand side is maximized.

    Args:
        x: A batch of inputs (PyTorch Tensor, TensorFlow Eager Tensor, NumPy
            Array, JAX Array, or EagerPy Tensor).
        delta: A batch of perturbation directions (same shape and type as x).
        eps: The target norm (non-negative float).
        a: The lower bound of the data domain.
        b: The upper bound of the data domain.

    Returns:
        eta: A batch of scales with the same number of dimensions as x but all
            axis == 1 except for the batch dimension.
    """
    (x, delta), restore_fn = ep.astensors_(x, delta)
    num_samples = x.shape[0]
    assert delta.shape[0] == num_samples
    rows = ep.arange(x, num_samples)

    delta_squared = delta.square().reshape((num_samples, -1))
    space = ep.where(delta >= 0, b - x, x - a).reshape((num_samples, -1))
    f_squared = space.square() / ep.maximum(delta_squared, 1e-20)
    ks = ep.argsort(f_squared, axis=-1)
    f_squared_sorted = f_squared[rows[:, ep.newaxis], ks]
    m = ep.cumsum(delta_squared[rows[:, ep.newaxis], ks.flip(axis=1)], axis=-1).flip(axis=1)
    dx = f_squared_sorted[:, 1:] - f_squared_sorted[:, :-1]
    dx = ep.concatenate((f_squared_sorted[:, :1], dx), axis=-1)
    dy = m * dx
    y = ep.cumsum(dy, axis=-1)
    c = y >= eps**2

    # Work-around to get first nonzero element in each row
    f = ep.arange(x, c.shape[-1], 0, -1)
    j = ep.argmax(c.astype(f.dtype) * f, axis=-1)

    eta_squared = f_squared_sorted[rows, j] - (y[rows, j] - eps**2) / m[rows, j]
    # It can happen that for certain rows even the largest j is not large enough
    # (i.e. c[:, -1] is False), then we will just use it (without any correction) as it's
    # the best we can do (this should also be the only cases where m[j] can be
    # 0 and they are thus not a problem)
    eta_squared = ep.where(c[:, -1], eta_squared, f_squared_sorted[:, -1])
    eta = ep.sqrt(eta_squared)
    eta = eta.reshape((-1,) + (1,) * (x.ndim - 1))

    return restore_fn(eta)