from typing import Optional

from .gradient_descent_base import L1BaseGradientDescent, L2BaseGradientDescent, LinfBaseGradientDescent, AdamOptimizer, Optimizer
import eagerpy as ep


class BasicIterativeAttackBase:
    """Base class for Basic Iterative Method attacks"""

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.2,
        abs_stepsize: Optional[float] = None,
        steps: int = 10,
        random_start: bool = False,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        raise NotImplementedError

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.2,
        abs_stepsize: Optional[float] = None,
        steps: int = 10,
        random_start: bool = False,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
    ):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        raise NotImplementedError


class L1BasicIterativeAttack(BasicIterativeAttackBase, L1BaseGradientDescent):
    """L1 Basic Iterative Method"""


class L2BasicIterativeAttack(BasicIterativeAttackBase, L2BaseGradientDescent):
    """L2 Basic Iterative Method"""


class LinfBasicIterativeAttack(BasicIterativeAttackBase, LinfBaseGradientDescent):
    """L-infinity Basic Iterative Method"""


class L1AdamBasicIterativeAttack(BasicIterativeAttackBase, L1BaseGradientDescent):
    """L1 Basic Iterative Method with Adam optimizer"""

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return AdamOptimizer(
            x,
            stepsize,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )


class L2AdamBasicIterativeAttack(BasicIterativeAttackBase, L2BaseGradientDescent):
    """L2 Basic Iterative Method with Adam optimizer"""

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return AdamOptimizer(
            x,
            stepsize,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )


class LinfAdamBasicIterativeAttack(BasicIterativeAttackBase, LinfBaseGradientDescent):
    """L-infinity Basic Iterative Method with Adam optimizer"""

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return AdamOptimizer(
            x,
            stepsize,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )