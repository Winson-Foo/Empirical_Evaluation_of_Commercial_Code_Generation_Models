class SqueezeTransform(Transform):
    """
    A transformation defined for image data that trades spatial dimensions for channel dimensions,
    i.e. "squeezes" the inputs along the channel dimensions.

    Implementation adapted from https://github.com/pclucas14/pytorch-glow and
    https://github.com/chaiyujin/glow-pytorch.

    Reference:
    > L. Dinh et al., Density estimation using Real NVP, ICLR 2017.
    """

    def __init__(self, factor=2):
        super(SqueezeTransform, self).__init__()
        self.validate_factor(factor)
        self.factor = factor

    def validate_factor(self, factor):
        if not isinstance(factor, int) or factor <= 1:
            raise ValueError("Factor must be an integer > 1.")

    def get_output_shape(self, channels, height, width):
        return (
            channels * self.factor * self.factor,
            height // self.factor,
            width // self.factor,
        )

    def validate_inputs(self, inputs):
        if inputs.dim() != 4:
            raise ValueError("Expecting inputs with 4 dimensions.")

    def validate_image_size(self, height, width):
        if height % self.factor != 0 or width % self.factor != 0:
            raise ValueError("Input image size not compatible with the factor.")

    def forward(self, inputs, context=None):
        self.validate_inputs(inputs)
        batch_size, channels, height, width = inputs.size()
        self.validate_image_size(height, width)

        split_height = height // self.factor
        split_width = width // self.factor

        inputs = inputs.view(
            batch_size,
            channels,
            split_height,
            self.factor,
            split_width,
            self.factor,
        )

        inputs = inputs.permute(0, 1, 3, 5, 2, 4).contiguous()

        inputs = inputs.view(
            batch_size,
            channels * self.factor * self.factor,
            split_height,
            split_width,
        )

        return inputs, inputs.new_zeros(batch_size)

    def inverse(self, inputs, context=None):
        self.validate_inputs(inputs)
        batch_size, channels, height, width = inputs.size()

        if channels < 4 or channels % 4 != 0:
            raise ValueError("Invalid number of channel dimensions.")

        split_channels = channels // self.factor ** 2

        inputs = inputs.view(
            batch_size,
            split_channels,
            self.factor,
            self.factor,
            height,
            width,
        )

        inputs = inputs.permute(0, 1, 4, 2, 5, 3).contiguous()

        inputs = inputs.view(
            batch_size,
            split_channels,
            height * self.factor,
            width * self.factor,
        )

        return inputs, inputs.new_zeros(batch_size)