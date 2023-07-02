from .gradient_descent_base import L1BaseGradientDescent, L2BaseGradientDescent, LinfBaseGradientDescent
from ..models.base import Model
from ..criteria import Misclassification, TargetedMisclassification
from .base import T
from typing import Union, Any

class FastGradientAttackBase:
    """Base class for Fast Gradient Method (FGM) attacks"""

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
        if hasattr(criterion, "target_classes"):
            raise ValueError("unsupported criterion")

        return super().run(
            model=model, inputs=inputs, criterion=criterion, epsilon=epsilon, **kwargs
        )


class L1FastGradientAttack(FastGradientAttackBase, L1BaseGradientDescent):
    """Fast Gradient Method (FGM) using the L1 norm"""

    def __init__(self, *, random_start: bool = False):
        super().__init__(random_start=random_start)


class L2FastGradientAttack(FastGradientAttackBase, L2BaseGradientDescent):
    """Fast Gradient Method (FGM)"""

    def __init__(self, *, random_start: bool = False):
        super().__init__(random_start=random_start)


class LinfFastGradientAttack(FastGradientAttackBase, LinfBaseGradientDescent):
    """Fast Gradient Sign Method (FGSM)"""

    def __init__(self, *, random_start: bool = False):
        super().__init__(random_start=random_start)