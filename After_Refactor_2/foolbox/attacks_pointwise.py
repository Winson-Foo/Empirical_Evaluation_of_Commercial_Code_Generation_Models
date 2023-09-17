import logging
from typing import Any, Callable, List, Optional, Tuple, Union

import eagerpy as ep
import numpy as np

from ..criteria import Criterion
from .base import FlexibleDistanceMinimizationAttack, MinimizationAttack, Model
from .base import get_criterion, get_is_adversarial, raise_if_kwargs, T, verify_input_bounds

class PointwiseAttack(FlexibleDistanceMinimizationAttack):
    """Starts with an adversarial and performs a binary search between
    the adversarial and the original for each dimension of the input
    individually. [#Sch18]_

    References:
        .. [#Sch18] Lukas Schott, Jonas Rauber, Matthias Bethge, Wieland Brendel,
               "Towards the first adversarially robust neural network model on MNIST",
               https://arxiv.org/abs/1805.09190
    """

    def __init__(self, init_attack: Optional[MinimizationAttack] = None, l2_binary_search: bool = True):
        self.init_attack = init_attack
        self.l2_binary_search = l2_binary_search

    def run(self, model: Model, inputs: T, criterion: Union[Criterion, Any] = None, **kwargs: Any) -> T:
        del kwargs

        x, restore_type = ep.astensor_(inputs)
        verify_input_bounds(x, model)
        criterion_ = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion_, model)
        starting_points = self._get_initial_points(x, criterion_, model)

        x_adv = ep.astensor(starting_points)
        self._check_adversarial(x_adv, is_adversarial)

        original_shape = x.shape

        found_index_to_manipulate = ep.ones(x, True)
        while ep.any(found_index_to_manipulate):
            x_adv, found_index_to_manipulate = self._manipulate_pixels(
                x, x_adv, found_index_to_manipulate, original_shape, is_adversarial
            )

        if self.l2_binary_search:
            x_adv = self._perform_binary_search(
                x, x_adv, original_shape, is_adversarial
            )

        return restore_type(x_adv)

    def _get_initial_points(self, x: ep.Tensor, criterion: Criterion, model: Model) -> ep.Tensor:
        if self.init_attack is None:
            init_attack = SaltAndPepperNoiseAttack()
            logging.info(
                f"Neither starting_points nor init_attack given. Falling"
                f" back to {init_attack!r} for initialization."
            )
        else:
            init_attack = self.init_attack
        return init_attack.run(model, x, criterion)

    def _check_adversarial(self, x_adv: ep.Tensor, is_adversarial: Callable) -> None:
        assert is_adversarial(x_adv).all()

    def _manipulate_pixels(
        self,
        x: ep.Tensor,
        x_adv: ep.Tensor,
        found_index_to_manipulate: ep.Tensor,
        original_shape: Tuple,
        is_adversarial: Callable,
    ) -> Tuple[ep.Tensor, ep.Tensor]:
        diff_mask = (ep.abs(x - x_adv) > 1e-8).numpy()
        diff_idxs = [z.nonzero()[0] for z in diff_mask]
        untouched_indices = [z.tolist() for z in diff_idxs]
        untouched_indices = [
            np.random.permutation(it).tolist() for it in untouched_indices
        ]

        found_index_to_manipulate = ep.from_numpy(x, np.zeros(len(x), dtype=bool))

        i = 0
        while i < max([len(it) for it in untouched_indices]):
            relevant_mask = [len(it) > i for it in untouched_indices]
            relevant_mask = np.array(relevant_mask, dtype=bool)
            relevant_mask_index = np.flatnonzero(relevant_mask)

            relevant_indices = [it[i] for it in untouched_indices if len(it) > i]

            old_values = x_adv[relevant_mask_index, relevant_indices]
            new_values = x[relevant_mask_index, relevant_indices]

            x_adv = ep.index_update(
                x_adv, (relevant_mask_index, relevant_indices), new_values
            )

            is_adv = is_adversarial(x_adv.reshape(original_shape))

            found_index_to_manipulate = ep.index_update(
                found_index_to_manipulate,
                relevant_mask_index,
                ep.logical_or(
                    found_index_to_manipulate, is_adv
                )[relevant_mask],
            )

            new_or_old_values = ep.where(is_adv, new_values, old_values)
            x_adv = ep.index_update(
                x_adv, (relevant_mask_index, relevant_indices), new_or_old_values
            )

            i += 1

        return x_adv, found_index_to_manipulate

    def _perform_binary_search(
        self,
        x: ep.Tensor,
        x_adv: ep.Tensor,
        original_shape: Tuple,
        is_adversarial: Callable,
    ) -> ep.Tensor:
        for _ in range(10):
            next_values = (x_adv + x) / 2

            x_adv = ep.index_update(x_adv, (mask_indices, indices), next_values)

            is_adv = is_adversarial(x_adv.reshape(original_shape))[mask]

            adv_values = ep.where(is_adv, next_values, adv_values)
            non_adv_values = ep.where(is_adv, non_adv_values, next_values)

        return adv_values