import math
from typing import Union, Optional, Any

import eagerpy as ep

from ..models import Model
from ..criteria import Misclassification, TargetedMisclassification
from ..distances import l2
from ..devutils import atleast_kd, flatten
from .base import MinimizationAttack
from .base import get_criterion
from .base import T
from .base import raise_if_kwargs
from .base import verify_input_bounds

def normalize_gradient_l2_norms(grad: ep.Tensor) -> ep.Tensor:
    grad = remove_zero_gradients(grad)
    grad = calculate_norms(grad)
    grad = normalize_norms(grad)
    return grad

def remove_zero_gradients(grad: ep.Tensor) -> ep.Tensor:
    norms = ep.norms.l2(flatten(grad), -1)
    grad = ep.where(atleast_kd(norms == 0, grad.ndim), ep.normal(grad, shape=grad.shape), grad)
    return grad

def calculate_norms(grad: ep.Tensor) -> ep.Tensor:
    norms = ep.norms.l2(flatten(grad), -1)
    norms = ep.maximum(norms, 1e-12)  # avoid division by zero
    return norms

def normalize_norms(grad: ep.Tensor) -> ep.Tensor:
    factor = 1 / norms
    factor = atleast_kd(factor, grad.ndim)
    return grad * factor

class DDNAttack(MinimizationAttack):
    distance = l2

    def __init__(
        self,
        *,
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
        early_stop: Optional[float] = None
    ) -> T:
        raise_if_kwargs()
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)

        verify_input_bounds(x, model)

        N = len(x)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(f"expected {name} to have shape ({N},), got {classes.shape}")

        max_stepsize = 1.0
        min_, max_ = model.bounds

        def loss_fn(
            inputs: ep.Tensor, labels: ep.Tensor
        ) -> Tuple[ep.Tensor, ep.Tensor]:
            logits = model(inputs)
            sign = -1.0 if targeted else 1.0
            loss = sign * ep.crossentropy(logits, labels).sum()
            return loss, logits

        grad_and_logits = ep.value_and_grad_fn(x, loss_fn, has_aux=True)

        delta = ep.zeros_like(x)

        epsilon = self.init_epsilon * ep.ones(x, len(x))
        worst_norm = ep.norms.l2(flatten(ep.maximum(x - min_, max_ - x)), -1)

        best_l2 = worst_norm
        best_delta = delta
        adv_found = ep.zeros(x, len(x)).bool()

        for i in range(self.steps):
            stepsize = get_stepsize(i)

            x_adv = x + delta

            _, logits, gradients = grad_and_logits(x_adv, classes)
            gradients = normalize_gradient_l2_norms(gradients)
            is_adversarial = criterion_(x_adv, logits)

            l2 = ep.norms.l2(flatten(delta), axis=-1)
            is_smaller = l2 <= best_l2

            is_both = ep.logical_and(is_adversarial, is_smaller)
            adv_found = ep.logical_or(adv_found, is_adversarial)
            best_l2 = ep.where(is_both, l2, best_l2)

            best_delta = ep.where(atleast_kd(is_both, x.ndim), delta, best_delta)

            delta = update_delta(delta, gradients, stepsize)

            epsilon = update_epsilon(epsilon, is_adversarial)
            epsilon = ep.minimum(epsilon, worst_norm)

            delta = project_to_epsilon_ball(delta, epsilon, x)

        x_adv = x + best_delta

        return restore_type(x_adv)

    def get_stepsize(self, step: int) -> float:
        return 0.01 + (self.max_stepsize - 0.01) * (1 + math.cos(math.pi * step / self.steps)) / 2

    def update_delta(self, delta: ep.Tensor, gradients: ep.Tensor, stepsize: float) -> ep.Tensor:
        return delta + stepsize * gradients

    def update_epsilon(self, epsilon: ep.Tensor, is_adversarial: ep.Tensor) -> ep.Tensor:
        return epsilon * ep.where(is_adversarial, 1.0 - self.gamma, 1.0 + self.gamma)

    def project_to_epsilon_ball(self, delta: ep.Tensor, epsilon: ep.Tensor, x: ep.Tensor) -> ep.Tensor:
        delta *= atleast_kd(epsilon / ep.norms.l2(flatten(delta), -1), x.ndim)
        delta = ep.clip(x + delta, *self.model.bounds) - x
        return delta