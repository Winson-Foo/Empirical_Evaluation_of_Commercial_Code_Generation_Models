import hydra
import torch

from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.decoder import GreedyDecoder, BeamCTCDecoder
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.model import DeepSpeech


def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    if loss_value == float("inf") or loss_value == float("-inf"):
        raise ValueError("WARNING: received an inf loss")
    elif torch.isnan(loss).sum() > 0:
        loss = torch.tensor(0.)
        print('WARNING: received a nan loss, setting loss value to 0')
    elif loss_value < 0:
        raise ValueError("WARNING: received a negative loss")
    return loss


def load_model(device, model_path):
    model = DeepSpeech.load_from_checkpoint(hydra.utils.to_absolute_path(model_path))
    model.eval()
    model = model.to(device)
    return model


def load_decoder(labels, cfg: LMConfig):
    if cfg.decoder_type == DecoderType.beam:
        if cfg.lm_path:
            cfg.lm_path = hydra.utils.to_absolute_path(cfg.lm_path)
        decoder = BeamCTCDecoder(labels=labels,
                                 lm_path=cfg.lm_path,
                                 alpha=cfg.alpha,
                                 beta=cfg.beta,
                                 cutoff_top_n=cfg.cutoff_top_n,
                                 cutoff_prob=cfg.cutoff_prob,
                                 beam_width=cfg.beam_width,
                                 num_processes=cfg.lm_workers,
                                 blank_index=labels.index('_'))
    else:
        decoder = GreedyDecoder(labels=labels,
                                blank_index=labels.index('_'))
    return decoder


def remove_parallel_wrapper(model):
    """
    Return the model or extract the model out of the parallel wrapper
    :param model: The training model
    :return: The model without parallel wrapper
    """
    # Take care of distributed/data-parallel wrapper
    return model.module if hasattr(model, "module") else model