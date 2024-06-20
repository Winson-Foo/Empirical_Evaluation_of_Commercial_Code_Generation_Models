from typing import Any, cast
import warnings
import eagerpy as ep
import torch
from ..types import BoundsInput, Preprocessing
from .base import ModelWithPreprocessing

def get_device(device: torch.device) -> torch.device:
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device

def _model(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.set_grad_enabled(x.requires_grad):
        result = cast(torch.Tensor, model(x))
    return result

class PyTorchModel(ModelWithPreprocessing):
    def __init__(
        self,
        model: torch.nn.Module,
        bounds: BoundsInput,
        device: torch.device = None,
        preprocessing: Preprocessing = None,
    ):
        if not isinstance(model, torch.nn.Module):
            raise ValueError("expected model to be a torch.nn.Module instance")

        if model.training:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn(
                    "The PyTorch model is in training mode and therefore might"
                    " not be deterministic. Call the eval() method to set it in"
                    " evaluation mode if this is not intended."
                )

        device = get_device(device)
        model = model.to(device)
        dummy = ep.zeros(0, device=device)

        super().__init__(
            lambda x: _model(model, x), bounds=bounds, dummy=dummy, preprocessing=preprocessing
        )

        self.data_format = "channels_first"
        self.device = device