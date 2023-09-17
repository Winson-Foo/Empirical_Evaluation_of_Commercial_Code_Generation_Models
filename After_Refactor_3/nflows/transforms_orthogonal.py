import torch
from torch import nn

from nflows.transforms.base import Transform
import nflows.utils.typechecks as check


def apply_transforms(inputs, q_vectors):
    """
    Apply the sequence of transforms parameterized by given q_vectors to inputs.

    Costs O(KDN), where:
    - K is number of transforms
    - D is dimensionality of inputs
    - N is number of inputs

    Args:
        inputs: Tensor of shape [N, D]
        q_vectors: Tensor of shape [K, D]

    Returns:
        A tuple of:
        - A Tensor of shape [N, D], the outputs.
        - A Tensor of shape [N], the log absolute determinants of the total transform.
    """
    squared_norms = torch.norm(q_vectors, dim=-1) ** 2
    outputs = inputs
    for q_vector, squared_norm in zip(q_vectors, squared_norms):
        temp = torch.matmul(outputs, q_vector.unsqueeze(-1))  # Inner product.
        temp = torch.ger(temp.squeeze(), (2.0 / squared_norm) * q_vector)  # Outer product.
        outputs = outputs - temp
    batch_size = inputs.shape[0]
    log_abs_det = torch.zeros(batch_size, dtype=inputs.dtype, device=inputs.device)
    return outputs, log_abs_det


class HouseholderSequence(Transform):
    """
    A sequence of Householder transforms.

    This class can be used as a way of parameterizing an orthogonal matrix.
    """

    def __init__(self, num_features, num_transforms):
        """
        Constructor.

        Args:
            num_features: int, dimensionality of the input.
            num_transforms: int, number of Householder transforms to use.

        Raises:
            TypeError: if arguments are not the right type.
        """
        if not check.is_positive_int(num_features):
            raise TypeError("Number of features must be a positive integer.")
        if not check.is_positive_int(num_transforms):
            raise TypeError("Number of transforms must be a positive integer.")

        super().__init__()
        self.num_features = num_features
        self.num_transforms = num_transforms

        q_vectors = torch.eye(num_transforms // 2, num_features).repeat(2, 1)
        if num_transforms % 2 != 0:  # odd number of transforms, including 1
            zero_vector = torch.zeros(1, num_features)
            zero_vector[0, num_transforms // 2] = 1
            q_vectors = torch.cat((q_vectors, zero_vector))
        self.q_vectors = nn.Parameter(q_vectors)

    def forward(self, inputs, context=None):
        return apply_transforms(inputs, self.q_vectors)

    def inverse(self, inputs, context=None):
        reverse_idx = torch.arange(self.num_transforms - 1, -1, -1)
        return apply_transforms(inputs, self.q_vectors[reverse_idx])

    def matrix(self):
        """
        Returns the orthogonal matrix that is equivalent to the total transform.

        Costs O(KD^2), where:
        - K is number of transforms
        - D is dimensionality of inputs

        Returns:
            A Tensor of shape [D, D].
        """
        identity = torch.eye(self.num_features, self.num_features)
        outputs, _ = self.inverse(identity)
        return outputs