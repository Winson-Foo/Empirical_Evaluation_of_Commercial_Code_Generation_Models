from typing import Union, Any

from ..models.base import Model
from ..criteria import Misclassification, TargetedMisclassification
from .base import T
from .gradient_descent_base import L1BaseGradientDescent, L2BaseGradientDescent, LinfBaseGradientDescent


class FastGradientAttackBase:
    """Base class for Fast Gradient Attack methods"""

    def __init__(self, *, random_start: bool = False):
        super().__init__(
            rel_stepsize=1.0,
            steps=1,
            random_start=random_start,
        )

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        return super().run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs
        )


class L1FastGradientAttack(FastGradientAttackBase, L1BaseGradientDescent):
    """Fast Gradient Method (FGM) using the L1 norm

    Args:
        random_start: Controls whether to randomly start within allowed epsilon ball.
    """


class L2FastGradientAttack(FastGradientAttackBase, L2BaseGradientDescent):
    """Fast Gradient Method (FGM) using the L2 norm

    Args:
        random_start: Controls whether to randomly start within allowed epsilon ball.
    """


class LinfFastGradientAttack(FastGradientAttackBase, LinfBaseGradientDescent):
    """Fast Gradient Sign Method (FGSM) using the Linf norm

    Args:
        random_start: Controls whether to randomly start within allowed epsilon ball.
    """