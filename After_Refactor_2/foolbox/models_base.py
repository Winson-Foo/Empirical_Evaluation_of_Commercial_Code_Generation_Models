from typing import List, Tuple, Dict, Optional
import copy
from abc import ABC, abstractmethod
import eagerpy as ep

class Model(ABC):
    @property
    @abstractmethod
    def bounds(self) -> Tuple[ep.TensorType, ep.TensorType]:
        pass

    @abstractmethod
    def __call__(self, inputs: ep.TensorType) -> ep.TensorType:
        pass

    def transform_bounds(self, bounds: Tuple[ep.TensorType, ep.TensorType]) -> "Model":
        return TransformBoundsWrapper(self, bounds)


class TransformBoundsWrapper(Model):
    def __init__(self, model: Model, bounds: Tuple[ep.TensorType, ep.TensorType]):
        self._model = model
        self._bounds = bounds

    @property
    def bounds(self) -> Tuple[ep.TensorType, ep.TensorType]:
        return self._bounds

    def __call__(self, inputs: ep.TensorType) -> ep.TensorType:
        x = ep.astensor(inputs)
        y = self._preprocess(x)
        z = self._model(y)
        return z

    def transform_bounds(self, bounds: Tuple[ep.TensorType, ep.TensorType], inplace: bool = False) -> Model:
        if inplace:
            self._bounds = bounds
            return self
        else:
            return TransformBoundsWrapper(copy.deepcopy(self._model), bounds)

    def _preprocess(self, inputs: ep.TensorType) -> ep.TensorType:
        if self.bounds == self._model.bounds:
            return inputs

        min_, max_ = self.bounds
        x = (inputs - min_) / (max_ - min_)

        model_min, model_max = self._model.bounds
        return x * (model_max - model_min) + model_min


class ModelWithPreprocessing(Model):
    def __init__(
        self,
        model: Callable[..., ep.TensorType],
        bounds: Tuple[ep.TensorType, ep.TensorType],
        dummy: ep.TensorType,
        preprocessing: Optional[Dict[str, ep.TensorType]] = None
    ):
        if not callable(model):
            raise ValueError("Expected model to be callable")

        self._model = model
        self._bounds = bounds
        self._dummy = dummy
        self._preprocess_args = self._process_preprocessing(preprocessing)

    @property
    def bounds(self) -> Tuple[ep.TensorType, ep.TensorType]:
        return self._bounds

    @property
    def dummy(self) -> ep.TensorType:
        return self._dummy

    def __call__(self, inputs: ep.TensorType) -> ep.TensorType:
        x = ep.astensor(inputs)
        y = self._preprocess(x)
        z = ep.astensor(self._model(y.raw))
        return z

    def transform_bounds(
        self,
        bounds: Tuple[ep.TensorType, ep.TensorType],
        inplace: bool = False,
        wrapper: bool = False
    ) -> Model:
        if wrapper and inplace:
            raise ValueError("Inplace and wrapper cannot both be True")

        if wrapper:
            return super().transform_bounds(bounds)

        if self.bounds == bounds:
            if inplace:
                return self
            else:
                return copy.deepcopy(self)

        model_min, model_max = self.bounds
        target_min, target_max = bounds
        scaling_factor = (target_max - target_min) / (model_max - model_min)

        mean, std, flip_axis = self._preprocess_args

        if mean is None:
            mean = ep.zeros(self._dummy, 1)
        mean = scaling_factor * (mean - model_min) + target_min

        if std is None:
            std = ep.ones(self._dummy, 1)
        std = scaling_factor * std

        if inplace:
            model = self
        else:
            model = copy.deepcopy(self)
        model._bounds = bounds
        model._preprocess_args = (mean, std, flip_axis)
        return model

    def _preprocess(self, inputs: ep.TensorType) -> ep.TensorType:
        mean, std, flip_axis = self._preprocess_args
        x = inputs
        if flip_axis is not None:
            x = x.flip(axis=flip_axis)
        if mean is not None:
            x = x - mean
        if std is not None:
            x = x / std
        assert x.dtype == inputs.dtype
        return x

    def _process_preprocessing(self, preprocessing: Optional[Dict[str, ep.TensorType]]) -> Tuple[
        Optional[ep.TensorType], Optional[ep.TensorType], Optional[int]]:
        if preprocessing is None:
            preprocessing = {}

        unsupported_keys = set(preprocessing.keys()) - {"mean", "std", "axis", "flip_axis"}
        if unsupported_keys:
            raise ValueError(f"Unknown preprocessing keys: {unsupported_keys}")

        mean = preprocessing.get("mean")
        std = preprocessing.get("std")
        axis = preprocessing.get("axis")
        flip_axis = preprocessing.get("flip_axis")

        def to_tensor(x: Optional[ep.TensorType]) -> Optional[ep.TensorType]:
            if x is None:
                return None
            return ep.astensor(x) if isinstance(x, ep.Tensor) else ep.from_numpy(self._dummy, x)

        mean = to_tensor(mean)
        std = to_tensor(std)

        def apply_axis(x: Optional[ep.TensorType], axis: int) -> Optional[ep.TensorType]:
            if x is None:
                return None
            if x.ndim != 1:
                raise ValueError("Non-None axis requires a 1D tensor")
            if axis >= 0:
                raise ValueError("Expected axis to be None or negative, -1 refers to the last axis")
            return ep.expand_dims(x, axis)

        if axis is not None:
            mean = apply_axis(mean, axis)
            std = apply_axis(std, axis)

        return mean, std, flip_axis