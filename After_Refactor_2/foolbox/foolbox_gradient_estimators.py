from typing import Callable, Tuple, Type
import eagerpy as ep
from .types import BoundsInput, Bounds
from .attacks.base import Attack


class GradientEstimator(Attack):  # type: ignore
    def __init__(self, attack_cls: Type[Attack], samples: int, sigma: float, bounds: BoundsInput, clip: bool):
        self.attack_cls = attack_cls
        self.samples = samples
        self.sigma = sigma
        self.bounds = Bounds(*bounds)
        self.clip = clip

    def __getattr__(self, name):
        return getattr(self.attack_cls, name)

    def __call__(self, *args, **kwargs):
        with self.attack_cls(*args, **kwargs) as attack:
            return self._value_and_grad(attack)

    def _value_and_grad(self, attack: Attack) -> Tuple[ep.Tensor, ep.Tensor]:
        loss_fn = attack.loss_fn
        x = attack.x

        value = loss_fn(x)

        gradient = ep.zeros_like(x)
        with ep.no_grad():
            for _ in range(self.samples // 2):
                noise = ep.normal(x, shape=x.shape)

                pos_theta = x + self.sigma * noise
                neg_theta = x - self.sigma * noise

                if self.clip:
                    pos_theta = pos_theta.clip(*self.bounds)
                    neg_theta = neg_theta.clip(*self.bounds)

                pos_loss = loss_fn(pos_theta)
                neg_loss = loss_fn(neg_theta)

                gradient += (pos_loss - neg_loss) * noise

            gradient /= 2 * self.sigma * 2 * self.samples

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
        raise ValueError("This attack does not support gradient estimators.")  # pragma: no cover

    class AttackWithESGradientEstimator(GradientEstimator, AttackCls):
        pass

    AttackWithESGradientEstimator.__name__ = AttackCls.__name__ + "WithESGradientEstimator"
    AttackWithESGradientEstimator.__qualname__ = AttackCls.__qualname__ + "WithESGradientEstimator"
    return AttackWithESGradientEstimator


es_gradient_estimator = evolutionary_strategies_gradient_estimator