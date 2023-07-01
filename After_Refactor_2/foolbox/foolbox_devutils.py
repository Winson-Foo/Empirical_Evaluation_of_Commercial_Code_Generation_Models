import eagerpy as ep


def flatten(tensor: ep.Tensor, keep_dims: int = 1) -> ep.Tensor:
    """
    Flatten a tensor while keeping specified number of dimensions.

    Args:
        tensor: The input tensor.
        keep_dims: The number of dimensions to keep (default is 1).

    Returns:
        The flattened tensor.
    """
    return tensor.flatten(start=keep_dims)


def atleast_kd(tensor: ep.Tensor, target_dims: int) -> ep.Tensor:
    """
    Ensure that the tensor has at least a specified number of dimensions.

    Args:
        tensor: The input tensor.
        target_dims: The desired number of dimensions.

    Returns:
        The tensor reshaped to have at least the specified number of dimensions.
    """
    shape = tensor.shape + (1,) * (k - tensor.ndim)
    return tensor.reshape(shape)