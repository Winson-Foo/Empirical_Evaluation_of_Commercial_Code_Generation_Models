from typing import Tuple, Optional

import torch
from torch import Tensor

from nflows.transforms.lu import LULinear
from nflows.transforms.permutations import RandomPermutation
from nflows.utils import torchutils


class OneByOneConvolution(LULinear):
    def __init__(self, num_channels: int, using_cache: bool = False, identity_init: bool = True):
        super().__init__(num_channels, using_cache, identity_init)
        self.permutation = RandomPermutation(num_channels, dim=1)

    def _lu_forward_inverse(self, inputs: Tensor, inverse: bool = False) -> Tuple[Tensor, Tensor]:
        batch_size, num_channels, height, width = inputs.shape
        inputs = inputs.permute(0, 2, 3, 1).reshape(batch_size * height * width, num_channels)

        if inverse:
            outputs, log_abs_det = super().inverse(inputs)
        else:
            outputs, log_abs_det = super().forward(inputs)

        outputs = outputs.reshape(batch_size, height, width, num_channels).permute(0, 3, 1, 2)
        log_abs_det = log_abs_det.reshape(batch_size, height, width)

        return outputs, torchutils.sum_except_batch(log_abs_det)

    def forward(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if inputs.dim() != 4:
            raise ValueError("Inputs must be a 4D tensor.")

        inputs, _ = self.permutation(inputs)

        return self._lu_forward_inverse(inputs, inverse=False)

    def inverse(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if inputs.dim() != 4:
            raise ValueError("Inputs must be a 4D tensor.")

        outputs, log_abs_det = self._lu_forward_inverse(inputs, inverse=True)

        outputs, _ = self.permutation.inverse(outputs)

        return outputs, log_abs_det