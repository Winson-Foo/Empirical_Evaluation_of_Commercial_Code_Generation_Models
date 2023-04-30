import json
from typing import Dict, List

import hydra
import torch
from torch.cuda.amp import autocast

from deepspeech_pytorch.configs.inference_config import TranscribeConfig
from deepspeech_pytorch.decoder import Decoder
from deepspeech_pytorch.loader.data_loader import ChunkSpectrogramParser
from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.utils import load_decoder, load_model


def load_assets(cfg: TranscribeConfig) -> Tuple[DeepSpeech, Decoder, ChunkSpectrogramParser]:
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )

    spect_parser = ChunkSpectrogramParser(
        audio_conf=model.spect_cfg,
        normalize=True
    )

    return model, decoder, spect_parser


def perform_inference(audio_path: str, params: Dict) -> Tuple[List, List]:
    model = params['model']
    decoder = params['decoder']
    spect_parser = params['spect_parser']
    device = params['device']
    precision = params['precision']

    hs = None
    all_outs = []
    with torch.no_grad():
        for spect in spect_parser.parse_audio(audio_path, params['chunk_size_seconds']):
            spect = spect.to(device)
            input_sizes = torch.IntTensor([spect.size(3)]).int()
            with autocast(enabled=precision == 16):
                out, output_sizes, hs = model(spect, input_sizes, hs)
            all_outs.append(out.cpu())
    all_outs = torch.cat(all_outs, axis=1)
    decoded_output, decoded_offsets = decoder.decode(all_outs)
    return decoded_output, decoded_offsets


def decode_results(decoded_output: List, decoded_offsets: List, cfg: TranscribeConfig) -> Dict:
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "path": cfg.model.model_path
            },
            "language_model": {
                "path": cfg.lm.lm_path
            },
            "decoder": {
                "alpha": cfg.lm.alpha,
                "beta": cfg.lm.beta,
                "type": cfg.lm.decoder_type.value,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(cfg.lm.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if cfg.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)
    return results


def transcribe(cfg: TranscribeConfig):
    model, decoder, spect_parser = load_assets(cfg)
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    params = {
        'model': model,
        'decoder': decoder,
        'spect_parser': spect_parser,
        'device': device,
        'precision': cfg.model.precision,
        'chunk_size_seconds': cfg.chunk_size_seconds
    }

    decoded_output, decoded_offsets = perform_inference(
        audio_path=hydra.utils.to_absolute_path(cfg.audio_path),
        params=params
    )
    results = decode_results(
        decoded_output=decoded_output,
        decoded_offsets=decoded_offsets,
        cfg=cfg
    )
    print(json.dumps(results))