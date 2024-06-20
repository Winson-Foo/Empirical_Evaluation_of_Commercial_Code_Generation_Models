from typing import Optional, Any, Tuple, Union
import math

import eagerpy as ep

from ..models import Model
from ..criteria import TargetedMisclassification
from ..distances import linf
from .base import FixedEpsilonAttack, get_channel_axis, raise_if_kwargs, T, verify_input_bounds
from .gen_attack_utils import rescale_images

class GenAttack(FixedEpsilonAttack):
    """A black-box algorithm for L-infinity adversarials. [#Alz18]_

    This attack is performs a genetic search in order to find an adversarial
    perturbation in a black-box scenario in as few queries as possible.

    References:
        .. [#Alz18] Moustafa Alzantot, Yash Sharma, Supriyo Chakraborty, Huan Zhang,
           Cho-Jui Hsieh, Mani Srivastava,
           "GenAttack: Practical Black-box Attacks with Gradient-Free
           Optimization",
           https://arxiv.org/abs/1805.11090
    """

    def __init__(
        self,
        *,
        steps: int = 1000,
        population: int = 10,
        mutation_probability: float = 0.10,
        mutation_range: float = 0.15,
        sampling_temperature: float = 0.3,
        channel_axis: Optional[int] = None,
        reduced_dims: Optional[Tuple[int, int]] = None,
    ):
        self.steps = steps
        self.population = population
        self.min_mutation_probability = mutation_probability
        self.min_mutation_range = mutation_range
        self.sampling_temperature = sampling_temperature
        self.channel_axis = channel_axis
        self.reduced_dims = reduced_dims

    distance = linf

    def apply_noise(
        self,
        x: ep.TensorType,
        noise: ep.TensorType,
        epsilon: float,
        channel_axis: Optional[int],
    ) -> ep.TensorType:
        if noise.shape != x.shape and channel_axis is not None:
            noise = rescale_images(noise, x.shape, channel_axis)

        noise = ep.clip(noise, -epsilon, +epsilon)
        return ep.clip(x + noise, 0.0, 1.0)

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: TargetedMisclassification,
        *,
        epsilon: float,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs

        verify_input_bounds(x, model)

        N = len(x)
        noise_shape: Union[Tuple[int, int, int, int], Tuple[int, ...]]
        channel_axis: Optional[int] = None

        if isinstance(criterion, TargetedMisclassification):
            classes = criterion.target_classes
        else:
            raise ValueError("unsupported criterion")

        if classes.shape != (N,):
            raise ValueError(
                f"expected target_classes to have shape ({N},), got {classes.shape}"
            )

        if self.reduced_dims is not None:
            noise_shape, channel_axis = self._get_reduced_dims_noise_shape(x, model)
        else:
            noise_shape = x.shape[1:]  # pragma: no cover

        num_plateaus = ep.zeros(x, len(x))
        mutation_probability = (
            ep.ones_like(num_plateaus) * self.min_mutation_probability
        )
        mutation_range = ep.ones_like(num_plateaus) * self.min_mutation_range

        noise_pops = ep.uniform(
            x, (N, self.population, *noise_shape), -epsilon, epsilon
        )

        for step in range(self.steps):
            fitness, is_adv = self._calculate_fitness_and_is_adv(
                x, noise_pops, model, classes, epsilon, channel_axis
            )

            elite_idxs = ep.argmax(fitness, 0)
            elite_noise = noise_pops[range(N), elite_idxs]
            is_adv = is_adv[range(N), elite_idxs]

            if is_adv.all():
                return restore_type(
                    self.apply_noise(x, elite_noise, epsilon, channel_axis)
                )

            probs = self._calculate_probs(fitness)
            parents_idxs = self._calculate_parents_idxs(
                probs, N, self.population, parents_idxs
            )

            noise_pops = self._create_new_noise_pops(
                noise_pops, parents_idxs, N, noise_shape, epsilon, mutation_range
            )

            n_its_wo_change = ep.where(elite_idxs == 0, n_its_wo_change + 1, ep.zeros_like(n_its_wo_change))
            num_plateaus = ep.where(n_its_wo_change >= 100, num_plateaus + 1, num_plateaus)
            n_its_wo_change = ep.where(n_its_wo_change >= 100, ep.zeros_like(n_its_wo_change), n_its_wo_change)
            mutation_probability = self._update_mutation_probability(
                num_plateaus, self.min_mutation_probability, mutation_probability
            )

            mutation_range = self._update_mutation_range(
                num_plateaus, self.min_mutation_range, mutation_range
            )

        return restore_type(self.apply_noise(x, elite_noise, epsilon, channel_axis))

    def _get_reduced_dims_noise_shape(
        self, x: ep.TensorType, model: Model
    ) -> Tuple[Union[Tuple[int, int, int, int], Tuple[int, ...]], Optional[int]]:
        # Implementation for getting noise shape and channel axis
        pass

    def _calculate_fitness_and_is_adv(
        self,
        x: ep.TensorType,
        noise_pops: ep.TensorType,
        model: Model,
        classes: ep.TensorType,
        epsilon: float,
        channel_axis: Optional[int],
    ) -> Tuple[ep.TensorType, ep.TensorType]:
        # Implementation for calculating fitness and is_adv
        pass

    def _calculate_probs(self, fitness: ep.TensorType) -> ep.TensorType:
        # Implementation for calculating probs
        pass

    def _calculate_parents_idxs(
        self,
        probs: ep.TensorType,
        N: int,
        population: int,
        parents_idxs: np.ndarray,
    ) -> np.ndarray:
        # Implementation for calculating parents_idxs
        pass

    def _create_new_noise_pops(
        self,
        noise_pops: ep.TensorType,
        parents_idxs: np.ndarray,
        N: int,
        noise_shape: Union[Tuple[int, int, int, int], Tuple[int, ...]],
        epsilon: float,
        mutation_range: ep.TensorType,
    ) -> ep.TensorType:
        # Implementation for creating new noise pops
        pass

    def _update_mutation_probability(
        self,
        num_plateaus: ep.TensorType,
        min_mutation_probability: float,
        mutation_probability: ep.TensorType,
    ) -> ep.TensorType:
        # Implementation for updating mutation probability
        pass

    def _update_mutation_range(
        self,
        num_plateaus: ep.TensorType,
        min_mutation_range: float,
        mutation_range: ep.TensorType,
    ) -> ep.TensorType:
        # Implementation for updating mutation range
        pass