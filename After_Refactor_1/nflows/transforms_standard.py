from typing import Iterable, Optional, Tuple, Union
import warnings

import torch
from torch import Tensor

from nflows.transforms.base import Transform


class IdentityTransform(Transform):
    """Transform that leaves input unchanged."""

    def forward(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size = inputs.size(0)
        logabsdet = inputs.new_zeros(batch_size)
        return inputs, logabsdet

    def inverse(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        return self.forward(inputs, context)


class PointwiseAffineTransform(Transform):
    """Forward transform X = X * scale + shift."""

    def __init__(
        self, shift: Union[Tensor, float] = 0.0, scale: Union[Tensor, float] = 1.0,
    ):
        super().__init__()
        self._shift = torch.as_tensor(shift)
        self._scale = torch.as_tensor(scale)

        if torch.any(self._scale == 0.0):
            raise ValueError("Scale must be non-zero.")

        self._log_abs_scale = torch.log(torch.abs(self._scale))

    def _batch_logabsdet(self, batch_shape: Iterable[int]) -> Tensor:
        """Return log abs det with input batch shape."""
        if self._log_abs_scale.dim() > 0:
            return self._log_abs_scale.expand(batch_shape).sum()
        else:
            return self._log_abs_scale * torch.prod(torch.tensor(batch_shape))

    def forward(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size, *batch_shape = inputs.size()

        outputs = inputs * self._scale + self._shift
        logabsdet = self._batch_logabsdet(batch_shape).expand(batch_size)

        return outputs, logabsdet

    def inverse(self, inputs: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        batch_size, *batch_shape = inputs.size()
        outputs = (inputs - self._shift) / self._scale
        logabsdet = -self._batch_logabsdet(batch_shape).expand(batch_size)

        return outputs, logabsdet


class AffineTransform(PointwiseAffineTransform):
    def __init__(
        self, shift: Union[Tensor, float] = 0.0, scale: Union[Tensor, float] = 1.0,
    ):
        warnings.warn("`AffineTransform` is deprecated; use `PointwiseAffineTransform` instead.", DeprecationWarning)
        super().__init__(shift, scale)


# Alias for backward compatibility.
AffineScalarTransform = AffineTransform