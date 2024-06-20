from typing import Union, Tuple, Optional
import eagerpy as ep

from ..models import Model
from ..criteria import Misclassification
from ..distances import l2
from ..devutils import atleast_kd, flatten
from .base import MinimizationAttack
from .base import get_criterion, raise_if_kwargs, T
from .base import verify_input_bounds

class NewtonFoolAttack(MinimizationAttack):
    """Implementation of the NewtonFool Attack. [#Jang17]_

    Args:
        steps : Number of update steps to perform.
        step_size : Size of each update step.

    References:
        .. [#Jang17] Uyeong Jang et al., "Objective Metrics and Gradient Descent
            Algorithms for Adversarial Examples in Machine Learning",
            https://dl.acm.org/citation.cfm?id=3134635
    """

    distance = l2

    def __init__(self, steps: int = 100, step_size: float = 0.01):
        self.steps = steps
        self.step_size = step_size

    def run(self, model: Model, inputs: T, criterion: Union[Misclassification, T],
            *, early_stop: Optional[float] = None) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        
        verify_input_bounds(x, model)
        
        num_instances = len(x)
        
        if isinstance(criterion_, Misclassification):
            target_labels = criterion_.labels
        else:
            raise ValueError("Unsupported criterion")
        
        if target_labels.shape != (num_instances,):
            raise ValueError(f"Expected labels to have shape ({num_instances},), got {target_labels.shape}")
        
        min_bound, max_bound = model.bounds
        
        x_norm_squared = flatten(x.square()).sum(axis=1)

        for _ in range(self.steps):
            loss, (scores, pred_scores) = self.get_loss(model, x, target_labels, num_instances)
            gradients = self.calculate_gradients(x, loss)
            pred_labels = scores.argmax(axis=-1)
            gradients_norm = flatten(gradients.square()).sum(axis=1)
            delta = self.calculate_delta(x_norm_squared, gradients_norm, pred_scores, num_instances)

            is_not_adversarial = (pred_labels == target_labels).float32()
            delta *= is_not_adversarial

            self.apply_perturbation(x, gradients, delta, gradients_norm, min_bound, max_bound)

        return restore_type(x)

    def get_loss(self, model: Model, x: ep.Tensor, target_labels: ep.Tensor, num_instances: int):
        def loss_fun(x: ep.Tensor) -> Tuple[ep.Tensor, Tuple[ep.Tensor, ep.Tensor]]:
            logits = model(x)
            scores = ep.softmax(logits)
            pred_scores = scores[range(num_instances), target_labels]
            loss = pred_scores.sum()
            return loss, (scores, pred_scores)

        return ep.value_aux_and_grad(loss_fun, x)

    def calculate_gradients(self, x: ep.Tensor, loss: ep.Tensor):
        gradients = ep.astensor(ep.gradient(loss, x))
        return gradients

    def calculate_delta(self, x_norm_squared: ep.Tensor, gradients_norm: ep.Tensor,
                        pred_scores: ep.Tensor, num_instances: int):
        a = self.step_size * x_norm_squared * gradients_norm
        b = pred_scores - 1.0 / num_instances
        delta = ep.minimum(a, b)
        return delta

    def apply_perturbation(self, x: ep.Tensor, gradients: ep.Tensor, delta: ep.Tensor,
                           gradients_norm: ep.Tensor, min_bound: ep.Tensor, max_bound: ep.Tensor):
        delta_scaled = atleast_kd(delta / gradients_norm.square(), gradients.ndim)
        x -= delta_scaled * gradients
        x = ep.clip(x, min_bound, max_bound)