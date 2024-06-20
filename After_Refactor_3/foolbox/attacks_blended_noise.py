from typing import Union, Optional, Any
import warnings
import numpy as np
import eagerpy as ep

from ..devutils import atleast_kd
from ..distances import Distance
from .base import (
    FlexibleDistanceMinimizationAttack,
    Model,
    Criterion,
    T,
    get_is_adversarial,
    get_criterion,
    raise_if_kwargs,
    verify_input_bounds,
)

class LinearSearchBlendedUniformNoiseAttack(FlexibleDistanceMinimizationAttack):
    def __init__(
        self,
        *,
        distance: Optional[Distance] = None,
        directions: int = 1000,
        steps: int = 1000,
    ):
        super().__init__(distance=distance)
        self.directions = directions
        self.steps = steps

        if directions <= 0:
            raise ValueError("directions must be larger than 0")

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, Any] = None,
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        verify_input_bounds(x, model)

        is_adversarial = get_is_adversarial(criterion_, model)
        min_bound, max_bound = model.bounds
        num_inputs = len(x)

        random_directions = self._generate_random_directions(x, min_bound, max_bound, num_inputs, is_adversarial)
        best_perturbation = self._find_best_perturbation(x, random_directions, is_adversarial)

        x = self._blend_input_with_perturbation(x, best_perturbation, random_directions)

        return restore_type(x)

    def _generate_random_directions(self, x: T, min_bound: float, max_bound: float, num_inputs: int, is_adversarial: T) -> T:
        random_directions = None

        for _ in range(self.directions):
            random_inputs = ep.uniform(x, x.shape, min_bound, max_bound)
            random_is_adversarial = atleast_kd(is_adversarial(random_inputs), x.ndim)

            if random_directions is None:
                random_directions = random_inputs
                random_is_adv = random_is_adversarial
            else:
                random_directions = ep.where(random_is_adv, random_directions, random_inputs)
                random_is_adv = random_is_adv.logical_or(random_is_adversarial)

            if random_is_adv.all():
                break

        if not random_is_adv.all():
            warning_message = (
                f"{self.__class__.__name__} failed to draw sufficient random "
                f"inputs that are adversarial ({random_is_adv.sum()} / {num_inputs})."
            )
            warnings.warn(warning_message)

        return random_directions

    def _find_best_perturbation(self, x: T, random_directions: T, is_adversarial: T) -> T:
        x0 = x
        epsilon_values = np.linspace(0, 1, num=self.steps + 1, dtype=np.float32)
        best_perturbation = ep.ones_like(x, (len(x),))

        for epsilon in epsilon_values:
            perturbed_inputs = (1 - epsilon) * x0 + epsilon * random_directions
            is_adversarial_perturbed = is_adversarial(perturbed_inputs)
            epsilon = epsilon.item()

            best_perturbation = ep.minimum(ep.where(is_adversarial_perturbed, epsilon, 1.0), best_perturbation)

            if (best_perturbation < 1).all():
                break

        return best_perturbation

    def _blend_input_with_perturbation(self, x: T, best_perturbation: T, random_directions: T) -> T:
        best_perturbation = atleast_kd(best_perturbation, x.ndim)
        perturbed_inputs = (1 - best_perturbation) * x + best_perturbation * random_directions
        return perturbed_inputs