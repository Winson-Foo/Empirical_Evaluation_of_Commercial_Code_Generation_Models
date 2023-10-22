import hydra
import torch

from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.decoder import GreedyDecoder, BeamCTCDecoder
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.model import DeepSpeech


def check_loss(loss):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    loss_value = loss.item()
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss_value):
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
        loss_value = 0
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error, loss_value


def load_model(device, model_path):
    model = DeepSpeech.load_from_checkpoint(hydra.utils.to_absolute_path(model_path), map_location=device)
    model.eval()
    model = model.to(device)
    return model


def load_decoder(labels, cfg: LMConfig):
    if cfg.decoder_type == DecoderType.beam:
        if cfg.lm_path:
            lm_path = hydra.utils.to_absolute_path(cfg.lm_path)
        else:
            raise ValueError("LM path not provided for beam decoder")
        decoder = BeamCTCDecoder(labels=labels, lm_path=lm_path, alpha=cfg.alpha,
                                 beta=cfg.beta, cutoff_top_n=cfg.cutoff_top_n, cutoff_prob=cfg.cutoff_prob,
                                 beam_width=cfg.beam_width, num_processes=cfg.lm_workers, blank_index=labels.index('_'))
    else:
        decoder = GreedyDecoder(labels=labels, blank_index=labels.index('_'))
    return decoder


def remove_parallel_wrapper(model):
    """
    Return the model or extract the model out of the parallel wrapper
    :param model: The training model
    :return: The model without parallel wrapper
    """
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_no_wrapper = model.module
    elif isinstance(model, torch.nn.DataParallel):
        model_no_wrapper = model.module
    else:
        model_no_wrapper = model
    return model_no_wrapper