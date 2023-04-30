import torch
from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.model import DeepSpeech


def is_valid_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training

    :param loss: The loss tensor
    :param loss_value: The loss value
    :return: Return if loss is valid, and the error in case it is not
    """
    if loss_value == float("inf") or loss_value == float("-inf"):
        return False, "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        return False, 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        return False, "WARNING: received a negative loss"
    else:
        return True, ""


def load_model(device, model_path):
    """
    Load the DeepSpeech model from checkpoint and move it to the specified device

    :param device: The device to move the model to
    :param model_path: The path to the model checkpoint file
    :return: The loaded model
    """
    model = DeepSpeech.load_from_checkpoint(model_path)
    model.eval()
    model = model.to(device)
    return model


def load_decoder(labels, lm_config):
    """
    Load the decoder based on the specified decoder type in the LMConfig

    :param labels: The list of labels
    :param lm_config: The LMConfig object
    :return: The decoder object
    """
    if lm_config.decoder_type == DecoderType.beam:
        from deepspeech_pytorch.decoder import BeamCTCDecoder
        if lm_config.lm_path:
            lm_config.lm_path = hydra.utils.to_absolute_path(lm_config.lm_path)
        decoder = BeamCTCDecoder(labels=labels,
                                 lm_path=lm_config.lm_path,
                                 alpha=lm_config.alpha,
                                 beta=lm_config.beta,
                                 cutoff_top_n=lm_config.cutoff_top_n,
                                 cutoff_prob=lm_config.cutoff_prob,
                                 beam_width=lm_config.beam_width,
                                 num_processes=lm_config.lm_workers,
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
    model_without_wrapper = model.module if hasattr(model, "module") else model
    return model_without_wrapper