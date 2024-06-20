from typing import Union, Optional, Any
import eagerpy as ep
import numpy as np

from ..models import Model
from ..criteria import Criterion
from ..distances import Distance
from .base import FlexibleDistanceMinimizationAttack
from .base import T
from .base import get_is_adversarial
from .base import get_criterion
from .base import raise_if_kwargs
from .base import verify_input_bounds

class BinarizationRefinementAttack(FlexibleDistanceMinimizationAttack):
    """For models that preprocess their inputs by binarizing the
    inputs, this attack can improve adversarials found by other
    attacks. It does this by utilizing information about the
    binarization and mapping values to the corresponding value in
    the clean input or to the right side of the threshold.

    Args:
        threshold : The threshold used by the models binarization. If none,
            defaults to (model.bounds()[1] - model.bounds()[0]) / 2.
        included_in : Whether the threshold value itself belongs to the lower or
            upper interval.
    """

    def __init__(
        self,
        *,
        distance: Optional[Distance] = None,
        threshold: Optional[float] = None,
        included_in: Union[Literal["lower"], Literal["upper"]] = "upper",
    ):
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
        del inputs, starting_points, kwargs

        verify_input_bounds(x, model)

        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)

        threshold = self._calculate_threshold(model)

        p = ep.full_like(o, ep.nan)

        lower, upper = self._calculate_intervals(threshold)

        p = self._map_values(p, o, x, lower, upper)

        self._check_threshold_match(is_adversarial, x, p)

        return restore_type(p)

    def _calculate_threshold(self, model: Model) -> float:
        if self.threshold is None:
            min_, max_ = model.bounds
            return (min_ + max_) / 2.0
        else:
            return self.threshold

    def _calculate_intervals(self, threshold: float) -> Tuple[float, float]:
        nptype = threshold.dtype.type
        offset = np.nextafter(nptype(1.0), nptype(2.0))
        
        if self.included_in == "lower":
            lower_ = threshold
            upper_ = np.nextafter(threshold, threshold + offset)
        elif self.included_in == "upper":
            lower_ = np.nextafter(threshold, threshold - offset)
            upper_ = threshold
        else:
            raise ValueError(
                f"expected included_in to be 'lower' or 'upper', found '{self.included_in}'"
            )

        assert lower_ < upper_

        lower = ep.ones_like(o) * lower_
        upper = ep.ones_like(o) * upper_

        return lower, upper

    def _map_values(
        self,
        p: T,
        o: T,
        x: T,
        lower: float,
        upper: float
    ) -> T:

        indices = ep.logical_and(o <= lower, x <= lower)
        p = ep.where(indices, o, p)

        indices = ep.logical_and(o <= lower, x >= upper)
        p = ep.where(indices, upper, p)

        indices = ep.logical_and(o >= upper, x <= lower)
        p = ep.where(indices, lower, p)

        indices = ep.logical_and(o >= upper, x >= upper)
        p = ep.where(indices, o, p)

        assert not ep.any(ep.isnan(p))

        return p

    def _check_threshold_match(
        self,
        is_adversarial: T,
        x: T,
        p: T
    ) -> None:
        is_adv1 = is_adversarial(x)
        is_adv2 = is_adversarial(p)
        if (is_adv1 != is_adv2).any():
            raise ValueError(
                "The specified threshold does not match what is done by the model."
            )