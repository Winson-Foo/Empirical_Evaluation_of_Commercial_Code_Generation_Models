from typing import Optional

import eagerpy as ep
from .gradient_descent_base import (
    L1BaseGradientDescent,
    L2BaseGradientDescent,
    LinfBaseGradientDescent,
    Optimizer,
    AdamOptimizer,
)


class BaseIterativeAttack:
    """Base class for iterative attacks"""

    def __init__(
        self,
        rel_stepsize: float = 0.2,
        abs_stepsize: Optional[float] = None,
        steps: int = 10,
        random_start: bool = False,
    ):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start


class L1BasicIterativeAttack(L1BaseGradientDescent, BaseIterativeAttack):
    """L1 Basic Iterative Method"""

    def __init__(self, *, rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )


class L2BasicIterativeAttack(L2BaseGradientDescent, BaseIterativeAttack):
    """L2 Basic Iterative Method"""

    def __init__(self, *, rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )


class LinfBasicIterativeAttack(LinfBaseGradientDescent, BaseIterativeAttack):
    """L-infinity Basic Iterative Method"""

    def __init__(self, *, rel_stepsize=0.2, abs_stepsize=None, steps=10, random_start=False):
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )


class BaseAdamIterativeAttack(BaseIterativeAttack):
    """Base class for iterative attacks with Adam optimizer"""

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
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon


class L1AdamBasicIterativeAttack(L1BaseGradientDescent, BaseAdamIterativeAttack):
    """L1 Basic Iterative Method with Adam optimizer"""

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
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon


class L2AdamBasicIterativeAttack(L2BaseGradientDescent, BaseAdamIterativeAttack):
    """L2 Basic Iterative Method with Adam optimizer"""

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
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon


class LinfAdamBasicIterativeAttack(LinfBaseGradientDescent, BaseAdamIterativeAttack):
    """L-infinity Basic Iterative Method with Adam optimizer"""

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
        super().__init__(
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
        )
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return AdamOptimizer(x, stepsize, self.adam_beta1, self.adam_beta2, self.adam_epsilon)