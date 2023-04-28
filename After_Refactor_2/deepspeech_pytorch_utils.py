import hydra
import torch
from typing import List
from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.decoder import GreedyDecoder, BeamCTCDecoder
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.model import DeepSpeech

def check_loss(loss: torch.Tensor, loss_value: float) -> Tuple[bool, str]:
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
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


def load_model(device: torch.device, model_path: str) -> DeepSpeech:
    """
    Load the DeepSpeech model from the given checkpoint path.
    Set the model to evaluation mode and move it to the given device.
    """
    model = DeepSpeech.load_from_checkpoint(hydra.utils.to_absolute_path(model_path))
    model.eval()
    model = model.to(device)
    return model


def load_beam_decoder(labels: List[str], cfg: LMConfig) -> BeamCTCDecoder:
    """
    Load the BeamCTCDecoder with the given configuration.
    """
    lm_path = hydra.utils.to_absolute_path(cfg.lm_path) if cfg.lm_path else None
    decoder = BeamCTCDecoder(labels=labels,
                             lm_path=lm_path,
                             alpha=cfg.alpha,
                             beta=cfg.beta,
                             cutoff_top_n=cfg.cutoff_top_n,
                             cutoff_prob=cfg.cutoff_prob,
                             beam_width=cfg.beam_width,
                             num_processes=cfg.lm_workers,
                             blank_index=labels.index('_'))
    return decoder


def load_greedy_decoder(labels: List[str]) -> GreedyDecoder:
    """
    Load the GreedyDecoder with the given labels.
    """
    decoder = GreedyDecoder(labels=labels,
                            blank_index=labels.index('_'))
    return decoder


def load_decoder(labels: List[str], cfg: LMConfig):
    """
    Load the appropriate decoder based on the given configuration.
    """
    if cfg.decoder_type == DecoderType.beam:
        return load_beam_decoder(labels, cfg)
    else:
        return load_greedy_decoder(labels)


def remove_parallel_wrapper(model):
    """
    Return the model or extract the model out of the parallel wrapper.
    """
    # Take care of distributed/data-parallel wrapper
    model_no_wrapper = model.module if hasattr(model, "module") else model
    return model_no_wrapper