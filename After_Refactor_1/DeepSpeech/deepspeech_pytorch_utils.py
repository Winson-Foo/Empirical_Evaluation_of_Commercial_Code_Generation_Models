# inference.py

import torch
from deepspeech_pytorch.decoder import GreedyDecoder, BeamCTCDecoder
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.model import DeepSpeech
from typing import Tuple


def load_model(device:str, model_path:str) -> DeepSpeech:
    """
    Load pre-trained DeepSpeech model from checkpoint
    :param device: Device (cpu or cuda) to load the model on
    :param model_path: Path to the checkpoint file
    :return: The loaded model
    """
    model = DeepSpeech.load_from_checkpoint(model_path)
    model.eval()
    model = model.to(device)
    return model


def remove_parallel_wrapper(model:torch.nn.Module) -> torch.nn.Module:
    """
    Extract the model out of the parallel wrapper, if present
    :param model: The model
    :return: The model without parallel wrapper
    """
    return model.module if hasattr(model, "module") else model


def check_loss(loss:torch.Tensor, loss_value:float) -> Tuple[bool, str]:
    """
    Check that loss is valid and will not break training
    :param loss: The loss tensor
    :param loss_value: The loss value
    :return: A tuple of boolean indicating whether loss is valid or not, and an error string (if any)
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error