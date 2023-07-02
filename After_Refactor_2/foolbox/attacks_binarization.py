from typing import Union, Optional, Any, List
from typing_extensions import Literal

import eagerpy as ep
import numpy as np

from ..models import Model
from ..criteria import Criterion
from ..distances import Distance
from .base import FlexibleDistanceMinimizationAttack, T
from .base import get_is_adversarial, get_criterion, raise_if_kwargs, verify_input_bounds


class BinarizationRefinementAttack(FlexibleDistanceMinimizationAttack):
    """
    For models that preprocess their inputs by binarizing the inputs,
    this attack can improve adversarials found by other attacks.
    It does this by utilizing information about the binarization and
    mapping values to the corresponding value in the clean input or
    to the right side of the threshold.

    Args:
        threshold: The threshold used by the models binarization.
            If none, defaults to (model.bounds()[1] - model.bounds()[0]) / 2.
        included_in: Whether the threshold value itself belongs to the lower or upper interval.
    """

    def __init__(
        self,
        distance: Optional[Distance] = None,
        threshold: Optional[float] = None,
        included_in: Union[Literal["lower"], Literal["upper"]] = "upper",
    ) -> None:
        super().__init__(distance=distance)
        self.threshold = threshold
        self.included_in = included_in

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Criterion, T],
        *,
        early_stop: Optional[float] = None,
        starting_points: Optional[T] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        if starting_points is None:
            raise ValueError("BinarizationRefinementAttack requires starting_points")
        
        (o, x), restore_type = ep.astensors_(inputs, starting_points)
        verify_input_bounds(x, model)

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        if self.threshold is None:
            min_, max_ = model.bounds
            threshold = (min_ + max_) / 2.0
        else:
            threshold = self.threshold

        assert o.dtype == x.dtype

        nptype = np.dtype(type(o.reshape(-1)[0].numpy()))
        if nptype not in [np.float16, np.float32, np.float64]:
            raise ValueError(
                f"expected dtype to be float16, float32 or float64, found '{nptype}'"
            )

        threshold = nptype(threshold)
        offset = nptype(1.0)

        if self.included_in == "lower":
            lower = threshold
            upper = np.nextafter(threshold, threshold + offset)
        elif self.included_in == "upper":
            lower = np.nextafter(threshold, threshold - offset)
            upper = threshold
        else:
            raise ValueError(
                f"expected included_in to be 'lower' or 'upper', found '{self.included_in}'"
            )

        assert lower < upper

        p = ep.full_like(o, ep.nan)

        lower_tensor = ep.ones_like(o) * lower
        upper_tensor = ep.ones_like(o) * upper

        indices = ep.logical_and(o <= lower_tensor, x <= lower_tensor)
        p = ep.where(indices, o, p)

        indices = ep.logical_and(o <= lower_tensor, x >= upper_tensor)
        p = ep.where(indices, upper_tensor, p)

        indices = ep.logical_and(o >= upper_tensor, x <= lower_tensor)
        p = ep.where(indices, lower_tensor, p)

        indices = ep.logical_and(o >= upper_tensor, x >= upper_tensor)
        p = ep.where(indices, o, p)

        assert not ep.any(ep.isnan(p))

        is_adv1 = is_adversarial(x)
        is_adv2 = is_adversarial(p)
        if (is_adv1 != is_adv2).any():
            raise ValueError(
                "The specified threshold does not match what is done by the model."
            )
        
        return restore_type(p)