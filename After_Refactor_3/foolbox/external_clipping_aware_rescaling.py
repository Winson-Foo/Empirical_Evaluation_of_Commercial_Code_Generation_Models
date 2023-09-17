import eagerpy as ep


def calculate_eta(x, delta, target_norm, lower_bound=0.0, upper_bound=1.0):
    """
    Calculates eta such that norm(clip(x + eta * delta, lower_bound, upper_bound) - x) == target_norm.

    Assumes x and delta have a batch dimension and target_norm, lower_bound, upper_bound are scalars. If the equation cannot be solved because target_norm is too large, the left hand side is maximized.

    Args:
        x: A batch of inputs (PyTorch Tensor, TensorFlow Eager Tensor, NumPy
            Array, JAX Array, or EagerPy Tensor).
        delta: A batch of perturbation directions (same shape and type as x).
        target_norm: The target norm (non-negative float).
        lower_bound: The lower bound of the data domain (float).
        upper_bound: The upper bound of the data domain (float).

    Returns:
        eta: A batch of scales with the same number of dimensions as x but all
            axis == 1 except for the batch dimension.
    """
    (x, delta), restore_fn = ep.astensors_(x, delta)
    batch_size = x.shape[0]
    assert delta.shape[0] == batch_size
    row_indices = ep.arange(x, batch_size)

    delta_square = delta.square().reshape((batch_size, -1))
    space = ep.where(delta >= 0, upper_bound - x, x - lower_bound).reshape((batch_size, -1))
    f2 = space.square() / ep.maximum(delta_square, 1e-20)
    ks = ep.argsort(f2, axis=-1)
    f2_sorted = f2[row_indices[:, ep.newaxis], ks]
    m = ep.cumsum(delta_square[row_indices[:, ep.newaxis], ks.flip(axis=1)], axis=-1).flip(axis=1)
    dx = f2_sorted[:, 1:] - f2_sorted[:, :-1]
    dx = ep.concatenate((f2_sorted[:, :1], dx), axis=-1)
    dy = m * dx
    y = ep.cumsum(dy, axis=-1)
    c = y >= target_norm ** 2

    # work-around to get first nonzero element in each row
    f = ep.arange(x, c.shape[-1], 0, -1)
    j = ep.argmax(c.astype(f.dtype) * f, axis=-1)

    eta2 = f2_sorted[row_indices, j] - (y[row_indices, j] - target_norm ** 2) / m[row_indices, j]
    # it can happen that for certain rows even the largest j is not large enough
    # (i.e. c[:, -1] is False), then we will just use it (without any correction) as it's
    # the best we can do (this should also be the only cases where m[j] can be
    # 0 and they are thus not a problem)
    eta2 = ep.where(c[:, -1], eta2, f2_sorted[:, -1])
    eta = ep.sqrt(eta2)
    eta = eta.reshape((-1,) + (1,) * (x.ndim - 1))

    return restore_fn(eta)