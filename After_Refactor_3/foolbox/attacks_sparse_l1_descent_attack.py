from typing import Optional
import numpy as np

from ..devutils import flatten, atleast_kd
from ..types import Bounds
from .gradient_descent_base import L1BaseGradientDescent, normalize_lp_norms

class SparseL1DescentAttack(L1BaseGradientDescent):
    def normalize(self, gradients, *, x, bounds):
        bad_pos = (
            (x == bounds.lower) & (gradients < 0) |
            (x == bounds.upper) & (gradients > 0)
        )
        gradients = np.where(bad_pos, np.zeros_like(gradients), gradients)

        abs_gradients = np.abs(gradients)
        quantiles = np.quantile(
            flatten(abs_gradients).numpy(), q=self.quantile, axis=-1
        )
        keep = abs_gradients >= atleast_kd(
            gradients.from_numpy(gradients, quantiles), gradients.ndim
        )
        e = np.where(keep, gradients.sign(), np.zeros_like(gradients))
        return normalize_lp_norms(e, p=1)

    def project(self, x, x0, epsilon):
        delta = flatten(x - x0)
        norms = delta.norms.l1(axis=-1)
        if (norms <= epsilon).all():
            return x

        n, d = delta.shape
        abs_delta = np.abs(delta)
        mu = -np.sort(-abs_delta, axis=-1)
        cumsums = mu.cumsum(axis=-1)
        js = 1.0 / np.arange(x, 1, d + 1).astype(x.dtype)
        temp = mu - js * (cumsums - epsilon)
        guarantee_first = np.arange(x, d).astype(x.dtype) / d
        rho = np.argmin((temp > 0).astype(x.dtype) + guarantee_first, axis=-1)
        theta = 1.0 / (1 + rho.astype(x.dtype)) * (cumsums[range(n), rho] - epsilon)
        delta = delta.sign() * np.maximum(abs_delta - theta[..., np.newaxis], 0)
        delta = delta.reshape(x.shape)
        return x0 + delta

    def __init__(
        self,
        quantile=0.99,
        rel_stepsize=0.2,
        abs_stepsize=None,
        steps=10,
        random_start=False,
    ):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )
        if not 0 <= quantile <= 1:
            raise ValueError(f"quantile needs to be between 0 and 1, got {quantile}")
        self.quantile = quantile