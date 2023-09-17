import torch
from nflows.transforms.base import Transform
from nflows.utils.typechecks import is_positive_int

class Permutation(Transform):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation: torch.Tensor, dim: int = 1):
        """
        Args:
            permutation: A 1D tensor representing the permutation.
            dim: The dimension along which to permute the inputs.
        """
        if permutation.dim() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not is_positive_int(dim):
            raise ValueError("dim must be a positive integer.")
            
        super().__init__()
        self.dim = dim
        self.register_buffer("permutation", permutation)

    @property
    def inverse_permutation(self):
        """The inverse of the permutation."""
        return torch.argsort(self.permutation)

    def permute(self, inputs: torch.Tensor, permutation: torch.Tensor, dim: int):
        """
        Permutes the inputs tensor along the specified dimension using the given permutation.

        Args:
            inputs: The input tensor to permute.
            permutation: The permutation tensor.
            dim: The dimension along which to permute the inputs.

        Returns:
            permuted_inputs: The permuted inputs tensor.
            logabsdet: The log absolute determinant of the Jacobian matrix.
        """
        if dim >= inputs.dim():
            raise ValueError(f"No dimension {dim} in inputs.")
        if inputs.size(dim) != len(permutation):
            raise ValueError(f"Dimension {dim} in inputs must be of size {len(permutation)}.")

        batch_size = inputs.size(0)
        permuted_inputs = torch.index_select(inputs, dim, permutation)
        logabsdet = inputs.new_zeros(batch_size)
        return permuted_inputs, logabsdet

    def forward(self, inputs: torch.Tensor, context=None):
        """Permutes the inputs tensor along the specified dimension using the stored permutation."""
        return self.permute(inputs, self.permutation, self.dim)

    def inverse(self, inputs: torch.Tensor, context=None):
        """Permutes the inputs tensor along the specified dimension using the inverse of the stored permutation."""
        return self.permute(inputs, self.inverse_permutation, self.dim)


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features: int, dim: int = 1):
        """
        Args:
            features: The number of features.
            dim: The dimension along which to permute the inputs.
        """
        if not is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.randperm(features), dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, features: int, dim: int = 1):
        """
        Args:
            features: The number of features.
            dim: The dimension along which to permute the inputs.
        """
        if not is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.arange(features - 1, -1, -1), dim)