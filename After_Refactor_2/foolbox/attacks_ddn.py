import math
from typing import Any, Tuple, Union, Optional

import eagerpy as ep
from eagerpy import Tensor

from ..criteria import Misclassification, TargetedMisclassification
from ..distances import l2
from ..models import Model
from .base import MinimizationAttack, get_criterion, raise_if_kwargs, T
from .base import verify_input_bounds

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
        *,
        init_epsilon: float = 1.0,
        steps: int = 100,
        gamma: float = 0.05,
    ) -> None:
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
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

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
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        delta = self._attack(model, x, classes, criterion_, targeted)

        x_adv = x + delta

        return restore_type(x_adv)

    def _attack(
        self,
        model: Model,
        x: Tensor,
        classes: Tensor,
        criterion: Union[Misclassification, TargetedMisclassification],
        targeted: bool
    ) -> Tensor:
        max_stepsize = 1.0
        min_, max_ = model.bounds

        grad_and_logits = ep.value_and_grad_fn(x, self._loss_fn, has_aux=True)

        delta = ep.zeros_like(x)

        epsilon = self.init_epsilon * ep.ones(x, len(x))
        worst_norm = self._get_worst_norm(x, min_, max_)

        best_l2 = worst_norm
        best_delta = delta
        adv_found = ep.zeros(x, len(x)).bool()

        for i in range(self.steps):
            stepsize = self._get_stepsize(i, max_stepsize)
            x_adv = self._get_x_adv(x, delta)

            _, logits, gradients = grad_and_logits(x_adv, classes)
            gradients = self._normalize_gradient_l2_norms(gradients)
            is_adversarial = criterion(x_adv, logits)

            l2 = ep.norms.l2(flatten(delta), axis=-1)
            is_smaller = l2 <= best_l2

            is_both = ep.logical_and(is_adversarial, is_smaller)
            adv_found = ep.logical_or(adv_found, is_adversarial)
            best_l2 = ep.where(is_both, l2, best_l2)

            best_delta = ep.where(atleast_kd(is_both, x.ndim), delta, best_delta)

            delta = self._update_delta(delta, stepsize, gradients)

            epsilon = self._update_epsilon(epsilon, is_adversarial)
            epsilon = ep.minimum(epsilon, worst_norm)

            delta = self._project_to_epsilon(delta, x, epsilon)

        return best_delta

    def _loss_fn(
        self,
        inputs: ep.Tensor,
        labels: ep.Tensor
    ) -> Tuple[ep.Tensor, ep.Tensor]:
        logits = model(inputs)

        sign = -1.0 if targeted else 1.0
        loss = sign * ep.crossentropy(logits, labels).sum()

        return loss, logits

    def _normalize_gradient_l2_norms(self, grad: Tensor) -> Tensor:
        norms = ep.norms.l2(flatten(grad), -1)

        grad = ep.where(
            atleast_kd(norms == 0, grad.ndim), ep.normal(grad, shape=grad.shape), grad
        )

        norms = ep.norms.l2(flatten(grad), -1)

        norms = ep.maximum(norms, 1e-12)
        factor = 1 / norms
        factor = atleast_kd(factor, grad.ndim)
        return grad * factor

    def _get_worst_norm(self, x: Tensor, min_: Tensor, max_: Tensor) -> Tensor:
        return ep.norms.l2(flatten(ep.maximum(x - min_, max_ - x)), -1)

    def _get_stepsize(self, i: int, max_stepsize: float) -> float:
        return 0.01 + (max_stepsize - 0.01) * (1 + math.cos(math.pi * i / self.steps)) / 2

    def _get_x_adv(self, x: Tensor, delta: Tensor) -> Tensor:
        return x + delta

    def _update_delta(self, delta: Tensor, stepsize: float, gradients: Tensor) -> Tensor:
        return delta + stepsize * gradients

    def _update_epsilon(self, epsilon: Tensor, is_adversarial: Tensor) -> Tensor:
        return epsilon * ep.where(is_adversarial, 1.0 - self.gamma, 1.0 + self.gamma)

    def _project_to_epsilon(self, delta: Tensor, x: Tensor, epsilon: Tensor) -> Tensor:
        return delta * atleast_kd(epsilon / ep.norms.l2(flatten(delta), -1), x.ndim)