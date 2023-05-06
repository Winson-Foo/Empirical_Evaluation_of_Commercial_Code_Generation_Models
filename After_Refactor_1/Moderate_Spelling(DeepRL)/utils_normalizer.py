import numpy as np
import torch

# Define a base normalizer that other normalizers can inherit from
class BaseNormalizer:
    def __init__(self, read_only=False):
        self.read_only = read_only

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False

    def state_dict(self):
        return None

    def load_state_dict(self, _):
        return

    # Implement __call__ method in the base class to be inherited
    def __call__(self, x):
        pass

# Define a mean std normalizer that inherits from the base normalizer
class MeanStdNormalizer(BaseNormalizer):
    """
    Normalizes input data using mean and standard deviation.
    """

    def __init__(self, read_only=False, clip=10.0, epsilon=1e-8):
        super().__init__(read_only)
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def __call__(self, x):
        """
        Normalizes input data using mean and standard deviation.

        :param x: Numpy array or PyTorch tensor to be normalized.
        :return: Normalized numpy array or PyTorch tensor.
        """
        if isinstance(x, np.ndarray):
            x = np.asarray(x)
        else:
            x = x.detach().cpu().numpy()

        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])

        if not self.read_only:
            self.rms.update(x)

        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon),
                       -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean,
                'var': self.rms.var}

    def load_state_dict(self, saved):
        self.rms.mean = saved['mean']
        self.rms.var = saved['var']

# Define a rescale normalizer that inherits from the base normalizer
class RescaleNormalizer(BaseNormalizer):
    """
    Rescales input data using a constant coefficient.
    """

    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef

    def __call__(self, x):
        """
        Rescales input data using a constant coefficient.

        :param x: Numpy array or PyTorch tensor to be rescaled.
        :return: Rescaled numpy array or PyTorch tensor.
        """
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)

        return self.coef * x

# Define an image normalizer that inherits from the rescale normalizer
class ImageNormalizer(RescaleNormalizer):
    """
    Normalizes input image data (in 0-255 range) using a rescale coefficient.
    """

    def __init__(self):
        super().__init__(1.0 / 255)

# Define a sign normalizer that inherits from the base normalizer
class SignNormalizer(BaseNormalizer):
    """
    Normalizes input data by taking the sign of each element.
    """

    def __call__(self, x):
        """
        Normalizes input data by taking the sign of each element.

        :param x: Numpy array or PyTorch tensor to be normalized.
        :return: Normalized numpy array or PyTorch tensor.
        """
        return np.sign(x)