import eagerpy as ep


def flatten_tensor(tensor: ep.Tensor, keep_dim: int = 1) -> ep.Tensor:
    """
    Flattens a tensor along the specified dimensions.

    Args:
        tensor: The input tensor to be flattened.
        keep_dim: The number of leading dimensions to keep.

    Returns:
        A flattened tensor.
    """
    return tensor.flatten(start=keep_dim)


def expand_tensor_dim(tensor: ep.Tensor, min_dim: int) -> ep.Tensor:
    """
    Expands the dimensions of a tensor to a minimum number of dimensions.

    Args:
        tensor: The input tensor to be expanded.
        min_dim: The minimum number of dimensions the tensor should have.

    Returns:
        A tensor with dimensions expanded to `min_dim`.
    """
    shape = tensor.shape + (1,) * (min_dim - tensor.ndim)
    return tensor.reshape(shape)