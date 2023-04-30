import logging
import os
from tempfile import NamedTemporaryFile

import torch
from flask import Flask, request, jsonify

from deepspeech_pytorch.configs.inference_config import ServerConfig
from deepspeech_pytorch.inference import run_transcribe
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.utils import load_model, load_decoder

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['.wav', '.mp3', '.ogg', '.webm'])


def load_server():
    cfg = ServerConfig()
    cfg.merge_with_yaml("config.yaml")
    logging.getLogger().setLevel(logging.DEBUG)
    device = torch.device("cuda" if cfg.model.cuda else "cpu")
    model = load_model(device=device, model_path=cfg.model.model_path)
    decoder = load_decoder(labels=model.labels, cfg=cfg.lm)
    spect_parser = SpectrogramParser(audio_conf=model.spect_cfg, normalize=True)
    return model, decoder, spect_parser, device, cfg.host, cfg.port


def is_file_allowed(filename):
    _, file_extension = os.path.splitext(filename)
    return file_extension.lower() in ALLOWED_EXTENSIONS


def transcribe_audio_file(audio_file, spect_parser, model, decoder, device):
    with NamedTemporaryFile(suffix=".wav") as tmp_saved_audio_file:
        audio_file.save(tmp_saved_audio_file.name)
        transcription, _ = run_transcribe(
            audio_path=tmp_saved_audio_file,
            spect_parser=spect_parser,
            model=model,
            decoder=decoder,
            device=device
        )
        return transcription


@app.route('/transcribe', methods=['POST'])
def transcribe_file():
    res = {}
    if 'file' not in request.files:
        res['status'] = "error"
        res['message'] = "audio file should be passed for the transcription."
        return jsonify(res)

    audio_file = request.files['file']
    if not is_file_allowed(audio_file.filename):
        res['status'] = "error"
        res['message'] = "{} is not supported format.".format(audio_file.filename)
        return jsonify(res)

    try:
        transcription = transcribe_audio_file(
            audio_file,
            spect_parser,
            model,
            decoder,
            device
        )
        res['status'] = "OK"
        res['transcription'] = transcription
    except Exception:
        res['status'] = "error"
        res['message'] = "Internal server error."

    return jsonify(res)


if __name__ == "__main__":
    model, decoder, spect_parser, device, host, port = load_server()
    app.run(host=host, port=port, debug=True, use_reloader=False)