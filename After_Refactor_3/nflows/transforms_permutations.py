import torch
from nflows.transforms.base import Transform
from nflows.utils.typechecks import is_positive_int

class Permutation(Transform):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation: torch.Tensor, dim: int = 1):
        if permutation.ndimension() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not is_positive_int(dim):
            raise ValueError("dim must be a positive integer.")
        super().__init__()
        self.dim = dim
        self.permutation = permutation

    @property
    def inverse_permutation(self):
        return torch.argsort(self.permutation)

    @staticmethod
    def permute(inputs: torch.Tensor, permutation: torch.Tensor, dim: int):
        if dim >= inputs.ndimension():
            raise ValueError("No dimension {} in inputs.".format(dim))
        if inputs.shape[dim] != len(permutation):
            raise ValueError(
                "Dimension {} in inputs must be of size {}.".format(dim, len(permutation))
            )
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs: torch.Tensor):
        return self.permute(inputs, self.permutation, self.dim)

    def inverse(self, inputs: torch.Tensor):
        return self.permute(inputs, self.inverse_permutation, self.dim)


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features: int, dim: int = 1):
        super().__init__(torch.randperm(features), dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, features: int, dim: int = 1):
        super().__init__(torch.arange(features - 1, -1, -1), dim)