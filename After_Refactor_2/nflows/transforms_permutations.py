import torch
from nflows.transforms.base import Transform
import nflows.utils.typechecks as check


class Permutation(Transform):
    """Permutes inputs on a given dimension using a given permutation."""

    def __init__(self, permutation: torch.Tensor, dim: int = 1):
        """
        Initialize the Permutation transform.

        Args:
            permutation: The permutation tensor.
            dim: The dimension to permute on.
        """
        if permutation.ndimension() != 1:
            raise ValueError("Permutation must be a 1D tensor.")
        if not check.is_positive_int(dim):
            raise ValueError("dim must be a positive integer.")

        super().__init__()
        self.dim = dim
        self.register_buffer("permutation", permutation)

    @property
    def inverse_permutation(self):
        """Get the inverse permutation tensor."""
        return torch.argsort(self.permutation)

    @staticmethod
    def permute(inputs: torch.Tensor, permutation: torch.Tensor, dim: int):
        """
        Permute the inputs tensor.

        Args:
            inputs: The inputs tensor to permute.
            permutation: The permutation tensor.
            dim: The dimension to permute on.

        Returns:
            outputs: The permuted tensor.
            logabsdet: The log absolute determinant.

        Raises:
            ValueError: If the specified dimension is not present in the inputs tensor.
            ValueError: If the size of the specified dimension is not equal to the length of the permutation tensor.
        """
        if dim >= inputs.ndimension():
            raise ValueError(f"No dimension {dim} in inputs.")
        if inputs.shape[dim] != len(permutation):
            raise ValueError(
                f"Dimension {dim} in inputs must be of size {len(permutation)}."
            )
        batch_size = inputs.shape[0]
        outputs = torch.index_select(inputs, dim, permutation)
        logabsdet = inputs.new_zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs: torch.Tensor, context=None):
        """
        Apply forward permutation.

        Args:
            inputs: The inputs tensor.
            context: Optional context tensor.

        Returns:
            outputs: The permuted tensor.
            logabsdet: The log absolute determinant.
        """
        return self.permute(inputs, self.permutation, self.dim)

    def inverse(self, inputs: torch.Tensor, context=None):
        """
        Apply inverse permutation.

        Args:
            inputs: The inputs tensor.
            context: Optional context tensor.

        Returns:
            outputs: The permuted tensor.
            logabsdet: The log absolute determinant.
        """
        return self.permute(inputs, self.inverse_permutation, self.dim)


class RandomPermutation(Permutation):
    """Permutes using a random, but fixed, permutation. Only works with 1D inputs."""

    def __init__(self, features: int, dim: int = 1):
        """
        Initialize the RandomPermutation transform.

        Args:
            features: The number of features.
            dim: The dimension to permute on.
        """
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.randperm(features), dim)


class ReversePermutation(Permutation):
    """Reverses the elements of the input. Only works with 1D inputs."""

    def __init__(self, features: int, dim: int = 1):
        """
        Initialize the ReversePermutation transform.

        Args:
            features: The number of features.
            dim: The dimension to permute on.
        """
        if not check.is_positive_int(features):
            raise ValueError("Number of features must be a positive integer.")
        super().__init__(torch.arange(features - 1, -1, -1), dim)