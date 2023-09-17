from abc import ABC, abstractmethod

import torch


class Transform(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def inverse(self, inputs):
        pass


class Linear(Transform):
    def __init__(self, features, bias=None):
        self.features = features
        self.bias = bias if bias is not None else torch.zeros(features)

    @abstractmethod
    def forward_no_cache(self, inputs):
        pass

    @abstractmethod
    def inverse_no_cache(self, inputs):
        pass

    def forward(self, inputs):
        return self.forward_no_cache(inputs)

    def inverse(self, inputs):
        return self.inverse_no_cache(inputs)

    @abstractmethod
    def weight(self):
        pass

    @abstractmethod
    def weight_inverse(self):
        pass

    @abstractmethod
    def logabsdet(self):
        pass