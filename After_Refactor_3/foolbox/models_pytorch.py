import warnings
from typing import Any, cast

import eagerpy as ep
import torch

from ..types import BoundsInput, Preprocessing
from .base import ModelWithPreprocessing


def get_device(device: Any) -> Any:
    """
    Get the torch device based on the input device.
    If the input device is None, return "cuda:0" if cuda is available, else return "cpu".
    If the input device is a string, return the corresponding torch device.
    Otherwise, return the input device.
    """
    if device is None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def initialize_model(model: torch.nn.Module, device: Any) -> torch.nn.Module:
    """
    Initialize the PyTorch model by moving it to the specified device.
    """
    return model.to(device)


def model_forward(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the PyTorch model.
    Ensures that the output only requires_grad if the input does.
    """
    with torch.set_grad_enabled(x.requires_grad):
        result = cast(torch.Tensor, model(x))
    return result


def initialize_pytorch_model(
    model: torch.nn.Module, bounds: BoundsInput, device: Any = None,
    preprocessing: Preprocessing = None
) -> PyTorchModel:
    """
    Initialize a PyTorch model with preprocessing and device configuration.
    """
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
    model = initialize_model(model, device)
    dummy = ep.torch.zeros(0, device=device)

    pytorch_model = ModelWithPreprocessing(
        model_forward, bounds=bounds, dummy=dummy, preprocessing=preprocessing
    )

    pytorch_model.data_format = "channels_first"
    pytorch_model.device = device

    return pytorch_model