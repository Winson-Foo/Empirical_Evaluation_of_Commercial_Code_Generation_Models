from typing import Optional, Tuple
from abc import ABC, abstractmethod
from ..types import Bounds, BoundsInput, Preprocessing

class Model(ABC):
    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        ...

    @abstractmethod
    def __call__(self, inputs) -> Any:
        """Passes inputs through the model and returns the model's output"""
        ...

    def transform_bounds(self, bounds: BoundsInput) -> "Model":
        """Returns a new model with the desired bounds and updates the preprocessing accordingly"""
        return TransformBoundsWrapper(self, bounds)


class TransformBoundsWrapper(Model):
    def __init__(self, model: Model, bounds: BoundsInput):
        self.wrapped_model = model
        self.bounds = Bounds(*bounds)

    def __call__(self, inputs):
        x = inputs
        y = self._preprocess(x)
        z = self.wrapped_model(y)
        return z

    def transform_bounds(self, bounds: BoundsInput, inplace: bool = False) -> Model:
        if inplace:
            self.bounds = Bounds(*bounds)
            return self
        else:
            return TransformBoundsWrapper(self.wrapped_model, bounds)

    def _preprocess(self, inputs):
        if self.bounds == self.wrapped_model.bounds:
            return inputs

        min_, max_ = self.bounds
        x = (inputs - min_) / (max_ - min_)

        min_, max_ = self.wrapped_model.bounds
        return x * (max_ - min_) + min_


class ModelWithPreprocessing(Model):
    def __init__(
        self,
        model: Callable[..., Any],
        bounds: BoundsInput,
        dummy: Any,
        preprocessing: Preprocessing = None,
    ):
        if not callable(model):
            raise ValueError("expected model to be callable")

        self.model = model
        self.bounds = Bounds(*bounds)
        self.dummy = dummy
        self.preprocessing_args = self._process_preprocessing(preprocessing)

    def __call__(self, inputs):
        x = inputs
        y = self._preprocess(x)
        z = self.model(y)
        return z

    def transform_bounds(
        self,
        bounds: BoundsInput,
        inplace: bool = False,
        wrapper: bool = False,
    ) -> Model:
        if wrapper:
            if inplace:
                raise ValueError("inplace and wrapper cannot both be True")
            return super().transform_bounds(bounds)

        if self.bounds == bounds:
            if inplace:
                return self
            else:
                return copy.copy(self)

        a, b = self.bounds
        c, d = bounds
        f = (d - c) / (b - a)

        mean, std, flip_axis = self.preprocessing_args

        if mean is None:
            mean = ep.zeros(self.dummy, 1)
        mean = f * (mean - a) + c

        if std is None:
            std = ep.ones(self.dummy, 1)
        std = f * std

        if inplace:
            model = self
        else:
            model = copy.copy(self)
        model.bounds = Bounds(*bounds)
        model.preprocessing_args = (mean, std, flip_axis)
        return model

    def _preprocess(self, inputs):
        mean, std, flip_axis = self.preprocessing_args
        x = inputs
        if flip_axis is not None:
            x = x.flip(axis=flip_axis)
        if mean is not None:
            x = x - mean
        if std is not None:
            x = x / std
        assert x.dtype == inputs.dtype
        return x

    def _process_preprocessing(self, preprocessing: Preprocessing) -> Tuple:
        if preprocessing is None:
            preprocessing = dict()

        unsupported = set(preprocessing.keys()) - {"mean", "std", "axis", "flip_axis"}
        if len(unsupported) > 0:
            raise ValueError(f"unknown preprocessing key: {unsupported.pop()}")

        mean = preprocessing.get("mean", None)
        std = preprocessing.get("std", None)
        axis = preprocessing.get("axis", None)
        flip_axis = preprocessing.get("flip_axis", None)

        def to_tensor(x):
            if x is None:
                return None
            if isinstance(x, ep.Tensor):
                return x
            try:
                y = ep.astensor(x)
                if not isinstance(y, type(self.dummy)):
                    raise ValueError
                return y
            except ValueError:
                return ep.from_numpy(self.dummy, x)

        mean_ = to_tensor(mean)
        std_ = to_tensor(std)

        def apply_axis(x, axis):
            if x is None:
                return None
            if x.ndim != 1:
                raise ValueError(f"non-None axis requires a 1D tensor, got {x.ndim}D")
            if axis >= 0:
                raise ValueError(
                    "expected axis to be None or negative, -1 refers to the last axis"
                )
            return atleast_kd(x, -axis)

        if axis is not None:
            mean_ = apply_axis(mean_, axis)
            std_ = apply_axis(std_, axis)

        return mean_, std_, flip_axis