from typing import Optional

from .gradient_descent_base import L1BaseGradientDescent, L2BaseGradientDescent, LinfBaseGradientDescent
from .gradient_descent_base import AdamOptimizer, Optimizer
import eagerpy as ep


class BasicIterativeAttackMixin:
    """Mixin for Basic Iterative Method"""

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.2,
        abs_stepsize: Optional[float] = None,
        steps: int = 10,
        random_start: bool = False,
    ):
        super().__init__(rel_stepsize=rel_stepsize, abs_stepsize=abs_stepsize, steps=steps, random_start=random_start)


class L1BasicIterativeAttack(BasicIterativeAttackMixin, L1BaseGradientDescent):
    """L1 Basic Iterative Method"""


class L2BasicIterativeAttack(BasicIterativeAttackMixin, L2BaseGradientDescent):
    """L2 Basic Iterative Method"""


class LinfBasicIterativeAttack(BasicIterativeAttackMixin, LinfBaseGradientDescent):
    """L-infinity Basic Iterative Method"""


class AdamBasicIterativeAttackMixin(BasicIterativeAttackMixin):
    """Mixin for Basic Iterative Method with Adam optimizer"""

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


class L1AdamBasicIterativeAttack(AdamBasicIterativeAttackMixin, L1BaseGradientDescent):
    """L1 Basic Iterative Method with Adam optimizer"""


class L2AdamBasicIterativeAttack(AdamBasicIterativeAttackMixin, L2BaseGradientDescent):
    """L2 Basic Iterative Method with Adam optimizer"""


class LinfAdamBasicIterativeAttack(AdamBasicIterativeAttackMixin, LinfBaseGradientDescent):
    """L-infinity Basic Iterative Method with Adam optimizer"""

    def get_optimizer(self, x: ep.Tensor, stepsize: float) -> Optimizer:
        return AdamOptimizer(
            x,
            stepsize,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon,
        )