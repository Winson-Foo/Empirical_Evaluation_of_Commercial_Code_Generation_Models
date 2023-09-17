from typing import Optional, Any, Tuple, Union
import math

import numpy as np
import eagerpy as ep

from ..models import Model
from ..criteria import TargetedMisclassification
from ..distances import linf
from .base import FixedEpsilonAttack
from .base import T
from .base import get_channel_axis
from .base import raise_if_kwargs
from .base import verify_input_bounds
from .gen_attack_utils import rescale_images


class GenAttack(FixedEpsilonAttack):
    """
    A black-box algorithm for L-infinity adversarials. [#Alz18]_

    This attack performs a genetic search to find an adversarial perturbation
    in a black-box scenario in as few queries as possible.

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
        classes = criterion.target_classes

        if classes.shape != (N,):
            raise ValueError(
                f"expected target_classes to have shape ({N},), got {classes.shape}"
            )

        noise_shape, channel_axis = self._get_noise_shape_and_channel_axis(x, model)

        num_plateaus = ep.zeros(x, len(x))
        mutation_probability = ep.ones_like(num_plateaus) * self.min_mutation_probability
        mutation_range = ep.ones_like(num_plateaus) * self.min_mutation_range

        noise_pops = ep.uniform(
            x, (N, self.population, *noise_shape), -epsilon, epsilon
        )

        for step in range(self.steps):
            fitness = self._calculate_fitness(model, x, noise_pops, classes)

            if self._check_adv_complete(fitness):
                return restore_type(self.apply_noise(x, elite_noise, epsilon, channel_axis))

            probs = self._get_probs(fitness)
            parents_idxs = self._get_parents_idxs(N, self.population, probs)

            new_noise_pops = [elite_noise]
            for i in range(self.population - 1):
                parents_1 = noise_pops[range(N), parents_idxs[2 * i]]
                parents_2 = noise_pops[range(N), parents_idxs[2 * i + 1]]

                # calculate crossover
                p = probs[parents_idxs[2 * i], range(N)] / (
                    probs[parents_idxs[2 * i], range(N)]
                    + probs[parents_idxs[2 * i + 1], range(N)]
                )
                p = ep.atleast_kd(p, x.ndim)
                p = ep.tile(p, (1, *noise_shape))

                crossover_mask = ep.uniform(p, p.shape, 0, 1) < p
                children = ep.where(crossover_mask, parents_1, parents_2)

                # calculate mutation
                mutations = ep.stack(
                    [
                        ep.uniform(
                            x,
                            noise_shape,
                            -mutation_range[i].item() * epsilon,
                            mutation_range[i].item() * epsilon,
                        )
                        for i in range(N)
                    ],
                    0,
                )

                mutation_mask = ep.uniform(children, children.shape)
                mutation_mask = mutation_mask <= ep.atleast_kd(
                    mutation_probability, children.ndim
                )
                children = ep.where(mutation_mask, children + mutations, children)

                # project back to epsilon range
                children = ep.clip(children, -epsilon, epsilon)

                new_noise_pops.append(children)

            noise_pops = ep.stack(new_noise_pops, 1)

            n_its_wo_change = self._get_num_iterations_wo_change(elite_idxs, n_its_wo_change)
            num_plateaus = self._update_plateaus(n_its_wo_change, num_plateaus)
            mutation_probability = self._update_mutation_probability(num_plateaus)
            mutation_range = self._update_mutation_range(num_plateaus)

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

    def _get_noise_shape_and_channel_axis(self, x, model):
        if self.reduced_dims is not None:
            if x.ndim != 4:
                raise NotImplementedError(
                    "only implemented for inputs with two spatial dimensions"
                    " (and one channel and one batch dimension)"
                )

            if self.channel_axis is None:
                maybe_axis = get_channel_axis(model, x.ndim)
                if maybe_axis is None:
                    raise ValueError(
                        "cannot infer the data_format from the model, please"
                        " specify channel_axis when initializing the attack"
                    )
                else:
                    channel_axis = maybe_axis
            else:
                channel_axis = self.channel_axis % x.ndim

            if channel_axis == 1:
                noise_shape = (x.shape[1], *self.reduced_dims)
            elif channel_axis == 3:
                noise_shape = (*self.reduced_dims, x.shape[3])
            else:
                raise ValueError(
                    "expected 'channel_axis' to be 1 or 3, got {channel_axis}"
                )
        else:
            noise_shape = x.shape[1:]  # pragma: no cover

        return noise_shape, channel_axis

    def _calculate_fitness(self, model, x, noise_pops, classes):
        fitness_l, is_adv_l = [], []

        for i in range(self.population):
            it = self.apply_noise(x, noise_pops[:, i], epsilon, channel_axis)
            logits = model(it)
            f = self._calculate_fitness(logits)
            a = self._is_adversarial(logits)
            fitness_l.append(f)
            is_adv_l.append(a)

        fitness = ep.stack(fitness_l)
        is_adv = ep.stack(is_adv_l, 1)
        elite_idxs = ep.argmax(fitness, 0)

        elite_noise = noise_pops[range(N), elite_idxs]
        is_adv = is_adv[range(N), elite_idxs]

        return fitness, is_adv, elite_idxs, elite_noise

    def _check_adv_complete(self, fitness):
        return is_adv.all()

    def _get_probs(self, fitness):
        return ep.softmax(fitness / self.sampling_temperature, 0)

    def _get_parents_idxs(self, N, population, probs):
        return np.stack(
            [
                np.random.choice(
                    self.population,
                    2 * self.population - 2,
                    replace=True,
                    p=probs[:, i],
                )
                for i in range(N)
            ],
            1,
        )

    def _get_num_iterations_wo_change(self, elite_idxs, n_its_wo_change):
        return ep.where(
            elite_idxs == 0,
            n_its_wo_change + 1,
            ep.zeros_like(n_its_wo_change)
        )

    def _update_plateaus(self, n_its_wo_change, num_plateaus):
        return ep.where(
            n_its_wo_change >= 100,
            num_plateaus + 1,
            num_plateaus
        )

    def _update_mutation_probability(self, num_plateaus):
        return ep.maximum(
            self.min_mutation_probability,
            0.5 * ep.exp(math.log(0.9) * ep.ones_like(num_plateaus) * num_plateaus),
        )

    def _update_mutation_range(self, num_plateaus):
        return ep.maximum(
            self.min_mutation_range,
            0.4 * ep.exp(math.log(0.9) * ep.ones_like(num_plateaus) * num_plateaus),
        )

    def _calculate_fitness(self, logits):
        first = logits[range(N), classes]
        second = ep.log(ep.exp(logits).sum(1) - first)

        return first - second

    def _is_adversarial(self, logits):
        return ep.argmax(logits, 1) == classes