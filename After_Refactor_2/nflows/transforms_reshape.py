import torch

from nflows.transforms.base import Transform


class SqueezeTransform(Transform):
    """A transformation defined for image data that trades spatial dimensions for channel dimensions."""

    def __init__(self, factor: int = 2):
        super().__init__()

        if factor <= 1:
            raise ValueError("Factor must be an integer > 1.")

        self.factor = factor

    def get_output_shape(self, c: int, h: int, w: int) -> tuple:
        return (c * self.factor * self.factor, h // self.factor, w // self.factor)

    def forward(self, inputs: torch.Tensor, context=None) -> tuple:
        if inputs.dim() != 4:
            raise ValueError("Expecting inputs with 4 dimensions")

        batch_size, c, h, w = inputs.size()

        if h % self.factor != 0 or w % self.factor != 0:
            raise ValueError("Input image size not compatible with the factor.")

        squeezed_inputs = inputs.view(
            batch_size,
            c,
            h // self.factor,
            self.factor,
            w // self.factor,
            self.factor
        )
        squeezed_inputs = squeezed_inputs.permute(0, 1, 3, 5, 2, 4).contiguous()
        squeezed_inputs = squeezed_inputs.view(
            batch_size,
            c * self.factor * self.factor,
            h // self.factor,
            w // self.factor,
        )

        return squeezed_inputs, torch.zeros(batch_size, device=inputs.device)

    def inverse(self, inputs: torch.Tensor, context=None) -> tuple:
        if inputs.dim() != 4:
            raise ValueError("Expecting inputs with 4 dimensions")

        batch_size, c, h, w = inputs.size()

        if c < 4 or c % 4 != 0:
            raise ValueError("Invalid number of channel dimensions.")

        unsqueezed_inputs = inputs.view(
            batch_size,
            c // self.factor ** 2,
            self.factor,
            self.factor,
            h,
            w
        )
        unsqueezed_inputs = unsqueezed_inputs.permute(0, 1, 4, 2, 5, 3).contiguous()
        unsqueezed_inputs = unsqueezed_inputs.view(
            batch_size,
            c // self.factor ** 2,
            h * self.factor,
            w * self.factor
        )

        return unsqueezed_inputs, torch.zeros(batch_size, device=inputs.device)