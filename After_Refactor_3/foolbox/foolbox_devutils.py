import eagerpy as ep

def flatten_tensor(tensor: ep.Tensor, start_dim: int = 1) -> ep.Tensor:
    return tensor.flatten(start=start_dim)

def expand_tensor_dimensions(tensor: ep.Tensor, target_num_dims: int) -> ep.Tensor:
    shape = tensor.shape + (1,) * (target_num_dims - tensor.ndim)
    return tensor.reshape(shape)