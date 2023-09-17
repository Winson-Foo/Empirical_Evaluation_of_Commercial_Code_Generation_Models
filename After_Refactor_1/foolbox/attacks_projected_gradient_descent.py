from typing import Optional
import eagerpy as ep
from .gradient_descent_base import L1BaseGradientDescent, AdamOptimizer, Optimizer
from .gradient_descent_base import L2BaseGradientDescent
from .gradient_descent_base import LinfBaseGradientDescent


class ProjectedGradientDescentAttackBase:
    """Projected Gradient Descent Attack Base Class"""

    def __init__(
        self,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
    ):
        self.rel_stepsize = rel_stepsize
        self.abs_stepsize = abs_stepsize
        self.steps = steps
        self.random_start = random_start
        self.optimizer = self.get_optimizer()

    def get_optimizer(self) -> Optimizer:
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


class L1ProjectedGradientDescentAttack(L1BaseGradientDescent, ProjectedGradientDescentAttackBase):
    """L1 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps: Number of update steps to perform.
        random_start: Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True
    ):
        L1BaseGradientDescent.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )
        ProjectedGradientDescentAttackBase.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )


class L2ProjectedGradientDescentAttack(L2BaseGradientDescent, ProjectedGradientDescentAttackBase):
    """L2 Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps: Number of update steps to perform.
        random_start: Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True
    ):
        L2BaseGradientDescent.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )
        ProjectedGradientDescentAttackBase.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )


class LinfProjectedGradientDescentAttack(LinfBaseGradientDescent, ProjectedGradientDescentAttackBase):
    """Linf Projected Gradient Descent

    Args:
        rel_stepsize: Stepsize relative to epsilon (defaults to 0.01 / 0.3).
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps: Number of update steps to perform.
        random_start: Whether the perturbation is initialized randomly or starts at zero.
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.01 / 0.3,
        abs_stepsize: Optional[float] = None,
        steps: int = 40,
        random_start: bool = True
    ):
        LinfBaseGradientDescent.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )
        ProjectedGradientDescentAttackBase.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )


class AdamProjectedGradientDescentAttack(ProjectedGradientDescentAttackBase):
    """Projected Gradient Descent with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps: Number of update steps to perform.
        random_start: Whether the perturbation is initialized randomly or starts at zero.
        adam_beta1: beta_1 parameter of Adam optimizer
        adam_beta2: beta_2 parameter of Adam optimizer
        adam_epsilon: epsilon parameter of Adam optimizer responsible for numerical stability
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8
    ):
        ProjectedGradientDescentAttackBase.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

    def get_optimizer(self) -> AdamOptimizer:
        return AdamOptimizer(
            self.x,
            self.stepsize,
            self.adam_beta1,
            self.adam_beta2,
            self.adam_epsilon
        )


class L1AdamProjectedGradientDescentAttack(L1BaseGradientDescent, AdamProjectedGradientDescentAttack):
    """L1 Projected Gradient Descent with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps: Number of update steps to perform.
        random_start: Whether the perturbation is initialized randomly or starts at zero.
        adam_beta1: beta_1 parameter of Adam optimizer
        adam_beta2: beta_2 parameter of Adam optimizer
        adam_epsilon: epsilon parameter of Adam optimizer responsible for numerical stability
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8
    ):
        L1BaseGradientDescent.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )
        AdamProjectedGradientDescentAttack.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon
        )


class L2AdamProjectedGradientDescentAttack(L2BaseGradientDescent, AdamProjectedGradientDescentAttack):
    """L2 Projected Gradient Descent with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps: Number of update steps to perform.
        random_start: Whether the perturbation is initialized randomly or starts at zero.
        adam_beta1: beta_1 parameter of Adam optimizer
        adam_beta2: beta_2 parameter of Adam optimizer
        adam_epsilon: epsilon parameter of Adam optimizer responsible for numerical stability
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8
    ):
        L2BaseGradientDescent.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )
        AdamProjectedGradientDescentAttack.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon
        )


class LinfAdamProjectedGradientDescentAttack(LinfBaseGradientDescent, AdamProjectedGradientDescentAttack):
    """Linf Projected Gradient Descent with Adam optimizer

    Args:
        rel_stepsize: Stepsize relative to epsilon
        abs_stepsize: If given, it takes precedence over rel_stepsize.
        steps: Number of update steps to perform.
        random_start: Whether the perturbation is initialized randomly or starts at zero.
        adam_beta1: beta_1 parameter of Adam optimizer
        adam_beta2: beta_2 parameter of Adam optimizer
        adam_epsilon: epsilon parameter of Adam optimizer responsible for numerical stability
    """

    def __init__(
        self,
        *,
        rel_stepsize: float = 0.025,
        abs_stepsize: Optional[float] = None,
        steps: int = 50,
        random_start: bool = True,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8
    ):
        LinfBaseGradientDescent.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start
        )
        AdamProjectedGradientDescentAttack.__init__(
            self,
            rel_stepsize=rel_stepsize,
            abs_stepsize=abs_stepsize,
            steps=steps,
            random_start=random_start,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon
        )