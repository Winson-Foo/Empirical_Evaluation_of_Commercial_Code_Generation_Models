from typing import Optional, Union, Tuple, Any
import eagerpy as ep

from ..models import Model
from ..criteria import Misclassification
from ..distances import l2
from ..devutils import atleast_kd, flatten
from .base import MinimizationAttack, get_criterion, raise_if_kwargs, verify_input_bounds
from .base import T


class NewtonFoolAttack(MinimizationAttack):
    """Implementation of the NewtonFool Attack. [#Jang17]_

    Args:
        steps: Number of update steps to perform.
        step_size: Size of each update step.

    References:
        .. [#Jang17] Uyeong Jang et al., "Objective Metrics and Gradient Descent
            Algorithms for Adversarial Examples in Machine Learning",
            https://dl.acm.org/citation.cfm?id=3134635
    """

    distance = l2

    def __init__(self, steps: int = 100, stepsize: float = 0.01):
        self.steps = steps
        self.stepsize = stepsize

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)

        verify_input_bounds(x, model)

        N = len(x)
        classes = criterion_.labels

        if classes.shape != (N,):
            raise ValueError(
                f"Expected labels to have shape ({N},), got {classes.shape}"
            )

        min_, max_ = model.bounds

        x_l2_norm = flatten(x.square()).sum(1)

        for i in range(self.steps):
            scores, pred_scores, gradients = self.calculate_scores_gradients(model, x)
            pred = scores.argmax(-1)
            num_classes = scores.shape[-1]

            gradients_l2_norm = flatten(gradients.square()).sum(1)

            delta = self.calculate_delta(pred_scores, gradients_l2_norm, x_l2_norm, num_classes)

            is_not_adversarial = (pred == classes).float32()
            delta *= is_not_adversarial

            x = self.apply_perturbation(x, gradients_l2_norm, delta, gradients, min_, max_)

        return restore_type(x)

    def calculate_scores_gradients(self, model: Model, x: ep.Tensor) -> Tuple[ep.Tensor, ep.Tensor, ep.Tensor]:
        def loss_fun(x: ep.Tensor) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            logits = model(x)
            scores = ep.softmax(logits)
            pred_scores = scores[range(len(x)), classes]
            loss = pred_scores.sum()
            return loss, (scores, pred_scores)

        return ep.value_aux_and_grad(loss_fun, x)[1]

    def calculate_delta(
        self,
        pred_scores: ep.Tensor,
        gradients_l2_norm: ep.Tensor,
        x_l2_norm: ep.Tensor,
        num_classes: int
    ) -> ep.Tensor:
        a = self.stepsize * x_l2_norm * gradients_l2_norm
        b = pred_scores - 1.0 / num_classes
        return ep.minimum(a, b)

    def apply_perturbation(
        self,
        x: ep.Tensor,
        gradients_l2_norm: ep.Tensor,
        delta: ep.Tensor,
        gradients: ep.Tensor,
        min_: ep.Tensor,
        max_: ep.Tensor
    ) -> ep.Tensor:
        a = atleast_kd(delta / gradients_l2_norm.square(), gradients.ndim)
        x -= a * gradients
        x = ep.clip(x, min_, max_)
        return x