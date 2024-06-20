from typing import Union, Optional, Any
from typing_extensions import Literal
import eagerpy as ep
import numpy as np

from ..models import Model
from ..criteria import Criterion
from ..distances import Distance
from .base import FlexibleDistanceMinimizationAttack, T, get_is_adversarial, get_criterion, raise_if_kwargs, verify_input_bounds


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

        threshold = self._get_threshold(model)
        p = self._refine_adversarial_points(o, x, threshold)

        is_adv1 = is_adversarial(x)
        is_adv2 = is_adversarial(p)
        if (is_adv1 != is_adv2).any():
            raise ValueError(
                "The specified threshold does not match what is done by the model."
            )
        return restore_type(p)

    def _get_threshold(self, model: Model) -> float:
        if self.threshold is None:
            min_, max_ = model.bounds
            threshold = (min_ + max_) / 2.0
        else:
            threshold = self.threshold
        return threshold

    def _refine_adversarial_points(self, o: T, x: T, threshold: float) -> T:
        assert o.dtype == x.dtype

        nptype = o.reshape(-1)[0].numpy().dtype.type
        if nptype not in [np.float16, np.float32, np.float64]:
            raise ValueError(
                f"expected dtype to be float16, float32 or float64, found '{nptype}'"
            )

        threshold = nptype(threshold)
        offset = nptype(1.0)

        lower, upper = self._get_bounds(threshold, offset, nptype)
        p = self._map_values(o, x, lower, upper)
        return p

    def _get_bounds(self, threshold: float, offset: float, nptype: type) -> Tuple[float, float]:
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
        return lower_, upper_

    def _map_values(self, o: T, x: T, lower: float, upper: float) -> T:
        p = ep.full_like(o, ep.nan)

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