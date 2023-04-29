import numpy as np

class GaussianProcess:
    """A Gaussian process for generating random noise."""

    def __init__(self, size: tuple, std: float) -> None:
        """Initialize the Gaussian process.

        Args:
            size (tuple): The size of the noise array to generate.
            std (float): The standard deviation of the noise.
        """
        self.size = size
        self.std = std

    def sample(self) -> np.ndarray:
        """Generate a sample of noise."""
        return np.random.randn(*self.size) * self.std()