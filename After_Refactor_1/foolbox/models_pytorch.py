import warnings
from typing import Any, cast

import eagerpy as ep
import torch

from ..types import BoundsInput, Preprocessing
from .base import ModelWithPreprocessing


def get_device(device: Any) -> Any:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


class PyTorchModel(ModelWithPreprocessing):
    def __init__(self, model: torch.nn.Module, bounds: BoundsInput, device: Any = None,
                 preprocessing: Preprocessing = None):
        self._validate_model(model)

        if model.training:
            self._warn_training_mode()

        device = get_device(device)
        model = model.to(device)
        dummy = ep.torch.zeros(0, device=device)
        super().__init__(_model=model, bounds=bounds, dummy=dummy, preprocessing=preprocessing)

        self.data_format = "channels_first"
        self.device = device

    @staticmethod
    def _validate_model(model: Any) -> None:
        if not isinstance(model, torch.nn.Module):
            raise ValueError("expected model to be a torch.nn.Module instance")

    @staticmethod
    def _warn_training_mode() -> None:
        warnings.warn(
            "The PyTorch model is in training mode and therefore might "
            "not be deterministic. Call the eval() method to set it in "
            "evaluation mode if this is not intended.",
            UserWarning
        )

    def _model(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(x.requires_grad):
            result = cast(torch.Tensor, self._model(x))
        return result