import eagerpy as ep

from ..criteria import Criterion
from ..models import Model
from .base import FlexibleDistanceMinimizationAttack
from .base import T, raise_if_kwargs, verify_input_bounds


class InversionAttack(FlexibleDistanceMinimizationAttack):
    """Creates "negative images" by inverting the pixel values. [#Hos16]_

    References:
        .. [#Hos16] Hossein Hosseini, Baicen Xiao, Mayoore Jaiswal, Radha Poovendran,
               "On the Limitation of Convolutional Neural Networks in Recognizing
               Negative Images",
               https://arxiv.org/abs/1607.02533
    """

    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Criterion = None,
        early_stop: float = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)

        x, restore_type = self._preprocess_inputs(inputs)
        self._verify_input_bounds(x, model)
        x = self._invert_pixel_values(x, model)
        return self._restore_inputs(x, restore_type)

    def _preprocess_inputs(self, inputs: T) -> Tuple[T, Callable]:
        x, restore_type = ep.astensor_(inputs)
        del inputs
        return x, restore_type

    def _verify_input_bounds(self, x: T, model: Model) -> None:
        min_, max_ = model.bounds
        verify_input_bounds(x, min_, max_)

    def _invert_pixel_values(self, x: T, model: Model) -> T:
        min_, max_ = model.bounds
        x = min_ + max_ - x
        return x

    def _restore_inputs(self, x: T, restore_type: Callable) -> T:
        return restore_type(x)