import hydra
from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.enums import DecoderType
from typing import List


def load_decoder(labels:List[str], cfg:LMConfig):
    """
    Load decoder (either GreedyDecoder or BeamCTCDecoder) based on decoder_type in LMConfig.
    :param labels: List of labels
    :param cfg: LMConfig object containing decoding hyperparameters
    :return: The loaded decoder object
    """
    if cfg.decoder_type == DecoderType.beam:
        from deepspeech_pytorch.decoder import BeamCTCDecoder
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