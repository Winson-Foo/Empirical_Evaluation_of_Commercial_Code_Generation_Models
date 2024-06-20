from typing import Callable, Tuple, Type
import eagerpy as ep
from .types import BoundsInput, Bounds
from .attacks.base import Attack


class GradientEstimator(Attack):  
    def value_and_grad(
        self,
        loss_fn: Callable[[ep.Tensor], ep.Tensor],
        x: ep.Tensor,
    ) -> Tuple[ep.Tensor, ep.Tensor]:
        value = loss_fn(x)

        gradient = ep.zeros_like(x)
        for _ in range(samples):
            noise = ep.normal(x, shape=x.shape)

            pos_theta = x + sigma * noise
            neg_theta = x - sigma * noise

            pos_theta_clipped = pos_theta.clip(*bounds) if clip else pos_theta
            neg_theta_clipped = neg_theta.clip(*bounds) if clip else neg_theta

            pos_loss = loss_fn(pos_theta_clipped)
            neg_loss = loss_fn(neg_theta_clipped)

            gradient += (pos_loss - neg_loss) * noise

        gradient /= sigma * samples

        return value, gradient


def evolutionary_strategies_gradient_estimator(
    AttackCls: Type[Attack],
    *,
    samples: int,
    sigma: float,
    bounds: BoundsInput,
    clip: bool,
) -> Type[Attack]:
    if not hasattr(AttackCls, "value_and_grad"):
        raise ValueError(
            "This attack does not support gradient estimators."
        )  # pragma: no cover

    bounds = Bounds(*bounds)

    GradientEstimator.__name__ = AttackCls.__name__ + "WithESGradientEstimator"
    GradientEstimator.__qualname__ = AttackCls.__qualname__ + "WithESGradientEstimator"
    return GradientEstimator