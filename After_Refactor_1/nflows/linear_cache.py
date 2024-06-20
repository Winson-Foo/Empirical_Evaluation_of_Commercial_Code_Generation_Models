"""
Helper class to store the cache of a linear transform.

The cache consists of: the weight matrix, its inverse and its log absolute determinant.
"""

import torch

class LinearCache:
    def __init__(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None

    def invalidate(self):
        self.weight = None
        self.inverse = None
        self.logabsdet = None