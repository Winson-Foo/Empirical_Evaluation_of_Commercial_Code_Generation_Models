import numpy as np
import torch
from baselines.common.running_mean_std import RunningMeanStd

class BaseNormalizer:
    """Base class for all normalizers in the codebase."""
    
    def __init__(self):
        self.read_only = False

    def set_read_only(self):
        self.read_only = True

    def unset_read_only(self):
        self.read_only = False
        
    def normalize(self, x):
        """Normalize input x."""
        raise NotImplementedError
        
    def state_dict(self):
        """Return state dict of the normalizer."""
        return None

    def load_state_dict(self, state_dict):
        """Load state dict into the normalizer."""
        pass

class MeanStdNormalizer(BaseNormalizer):
    """Normalizer that computes the mean and standard deviation of input x and normalizes it."""
    
    def __init__(self, clip=10.0, epsilon=1e-8):
        super().__init__()
        self.rms = None
        self.clip = clip
        self.epsilon = epsilon

    def normalize(self, x):
        x = np.asarray(x)
        if self.rms is None:
            self.rms = RunningMeanStd(shape=(1,) + x.shape[1:])
        if not self.read_only:
            self.rms.update(x)
        return np.clip((x - self.rms.mean) / np.sqrt(self.rms.var + self.epsilon), -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.rms.mean, 'var': self.rms.var}

    def load_state_dict(self, state_dict):
        self.rms.mean = state_dict['mean']
        self.rms.var = state_dict['var']

class RescaleNormalizer(BaseNormalizer):
    """Normalizer that rescales input x by a given coefficient."""
    
    def __init__(self, coef=1.0):
        super().__init__()
        self.coef = coef

    def normalize(self, x):
        if not isinstance(x, torch.Tensor):
            x = np.asarray(x)
        return self.coef * x

class ImageNormalizer(RescaleNormalizer):
    """Normalizer for image inputs that rescales the inputs by dividing by 255."""
    
    def __init__(self):
        super().__init__(1.0 / 255)

class SignNormalizer(BaseNormalizer):
    """Normalizer that returns the sign of input x."""
    
    def normalize(self, x):
        return np.sign(x)