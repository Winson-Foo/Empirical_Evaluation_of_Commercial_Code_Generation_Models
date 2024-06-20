import math
from typing import Union, Tuple, Optional, Any

import eagerpy as ep
from eagerpy import Tensor

from ..models import Model
from ..criteria import Misclassification, TargetedMisclassification
from ..distances import l2
from ..devutils import atleast_kd, flatten

from .base import MinimizationAttack
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from .base import verify_input_bounds


def normalize_gradient_l2_norms(grad: Tensor) -> Tensor:
    norms = ep.norms.l2(flatten(grad), -1)

    # remove zero gradients
    grad = ep.where(ep.atleast_kd(norms == 0, grad.ndim), ep.normal(grad, shape=grad.shape), grad)

    # calculate norms again for previously vanishing elements
    norms = ep.norms.l2(flatten(grad), -1)

    norms = ep.maximum(norms, 1e-12)  # avoid division by zero
    factor = 1 / norms
    factor = ep.atleast_kd(factor, grad.ndim)
    return grad * factor


class DDNAttack(MinimizationAttack):
    """The Decoupled Direction and Norm L2 adversarial attack. [#Rony18]_

    Args:
        init_epsilon : Initial value for the norm/epsilon ball.
        steps : Number of steps for the optimization.
        gamma : Factor by which the norm will be modified: new_norm = norm * (1 + or - gamma).

    References:
        .. [#Rony18] Jérôme Rony, Luiz G. Hafemann, Luiz S. Oliveira, Ismail Ben Ayed,
            Robert Sabourin, Eric Granger, "Decoupling Direction and Norm for
            Efficient Gradient-Based L2 Adversarial Attacks and Defenses",
            https://arxiv.org/abs/1811.09600
    """

    distance = l2

    def __init__(
        self,
        init_epsilon: float = 1.0,
        steps: int = 100,
        gamma: float = 0.05,
    ):
        self.init_epsilon = init_epsilon
        self.steps = steps
        self.gamma = gamma

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        self._verify_input_bounds(model, inputs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = self._get_criterion(criterion)

        N = len(x)
        classes = self._get_classes(criterion_, N, x.ndim)

        max_stepsize = 1.0
        min_, max_ = model.bounds

        delta = ep.zeros_like(x)
        epsilon = self.init_epsilon * ep.ones(x, len(x))
        worst_norm = ep.norms.l2(flatten(ep.maximum(x - min_, max_ - x)), -1)
        best_l2 = worst_norm
        best_delta = delta
        adv_found = ep.zeros(x, len(x)).bool()

        grad_and_logits = ep.value_and_grad_fn(x, self._loss_fn, has_aux=True)

        for i in range(self.steps):
            stepsize = self._get_stepsize(i)

            x_adv = x + delta
            _, logits, gradients = grad_and_logits(x_adv, classes)
            gradients = normalize_gradient_l2_norms(gradients)
            is_adversarial = criterion_(x_adv, logits)

            l2_norm = ep.norms.l2(flatten(delta), axis=-1)
            is_smaller = l2_norm <= best_l2

            is_both = ep.logical_and(is_adversarial, is_smaller)
            adv_found = ep.logical_or(adv_found, is_adversarial)
            best_l2 = ep.where(is_both, l2_norm, best_l2)
            best_delta = ep.where(ep.atleast_kd(is_both, x.ndim), delta, best_delta)

            delta = self._perform_step(delta, gradients, stepsize, x.ndim)
            epsilon = self._update_epsilon(epsilon, is_adversarial)
            epsilon = ep.minimum(epsilon, worst_norm)
            delta = self._project_to_epsilon_ball(delta, x, model.bounds)

        x_adv = x + best_delta

        return restore_type(x_adv)

    def _get_stepsize(self, i: int) -> float:
        return 0.01 + (self.max_stepsize - 0.01) * (1 + math.cos(math.pi * i / self.steps)) / 2

    def _perform_step(self, delta: Tensor, gradients: Tensor, stepsize: float, ndim: int) -> Tensor:
        delta += stepsize * gradients
        return delta

    def _update_epsilon(self, epsilon: Tensor, is_adversarial: Tensor) -> Tensor:
        factor = 1.0 - self.gamma if is_adversarial else 1.0 + self.gamma
        epsilon *= factor
        return epsilon

    def _project_to_epsilon_ball(self, delta: Tensor, x: Tensor, bounds: Tuple[float, float]) -> Tensor:
        delta *= ep.atleast_kd(self.epsilon / ep.norms.l2(flatten(delta), -1), x.ndim)
        delta = ep.clip(x + delta, *bounds) - x
        return delta

    @staticmethod
    def _get_criterion(criterion: Union[Misclassification, TargetedMisclassification, T]) -> T:
        criterion_ = get_criterion(criterion)

        if not isinstance(criterion_, (Misclassification, TargetedMisclassification)):
            raise ValueError("unsupported criterion")

        return criterion_

    @staticmethod
    def _get_classes(criterion: T, N: int, ndim: int) -> Tensor:
        classes = criterion.labels if isinstance(criterion, Misclassification) else criterion.target_classes

        if classes.shape != (N,):
            name = "target_classes" if isinstance(criterion, TargetedMisclassification) else "labels"
            raise ValueError(f"expected {name} to have shape ({N},), got {classes.shape}")

        return classes

    @staticmethod
    def _verify_input_bounds(model: Model, inputs: T) -> None:
        verify_input_bounds(inputs, model)

    @staticmethod
    def _loss_fn(inputs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        logits = model(inputs)
        sign = -1.0 if targeted else 1.0
        loss = sign * ep.crossentropy(logits, labels).sum()
        return loss, logits