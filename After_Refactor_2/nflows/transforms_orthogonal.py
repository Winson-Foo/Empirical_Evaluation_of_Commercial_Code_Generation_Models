import torch
from torch import nn

from nflows.transforms.base import Transform


class HouseholderSequence(Transform):
    def __init__(self, features: int, num_transforms: int):
        super().__init__()
        self.features = features
        self.num_transforms = num_transforms

        eye_matrix = torch.eye(num_transforms // 2, features)
        q_vectors = torch.cat((eye_matrix, torch.zeros(1, features)))
        q_vectors[-1, num_transforms // 2] = 1
        self.q_vectors = nn.Parameter(q_vectors)

    @staticmethod
    def _apply_transforms(inputs: torch.Tensor, q_vectors: torch.Tensor) -> torch.Tensor:
        squared_norms = torch.sum(q_vectors ** 2, dim=-1)
        outputs = inputs
        for q_vector, squared_norm in zip(q_vectors, squared_norms):
            inner_product = torch.einsum('bi, i -> b', outputs, q_vector)
            outer_product = torch.einsum('bi, b -> bi', inner_product, (2.0 / squared_norm) * q_vector)
            outputs = outputs - outer_product
        return outputs

    def forward(self, inputs: torch.Tensor, context=None) -> torch.Tensor:
        return self._apply_transforms(inputs, self.q_vectors)

    def inverse(self, inputs: torch.Tensor, context=None) -> torch.Tensor:
        reverse_idx = torch.arange(self.num_transforms - 1, -1, -1)
        return self._apply_transforms(inputs, self.q_vectors[reverse_idx])

    def matrix(self) -> torch.Tensor:
        identity = torch.eye(self.features, self.features)
        outputs = self.inverse(identity)[0]
        return outputs