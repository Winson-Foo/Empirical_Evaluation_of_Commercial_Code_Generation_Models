import logging
import os
from tempfile import NamedTemporaryFile

import hydra
import torch
from flask import Flask, request, jsonify
from hydra.core.config_store import ConfigStore

from deepspeech_pytorch.configs.inference_config import ServerConfig
from deepspeech_pytorch.inference import run_transcribe
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.utils import load_model, load_decoder

ALLOWED_EXTENSIONS = set(['.wav', '.mp3', '.ogg', '.webm'])
STATUS_OK = "OK"
STATUS_ERROR = "error"

app = Flask(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=ServerConfig)


def validate_file(file):
    """ Validates if the file is of allowed extension """
    res = {}
    filename = file.filename
    _, file_extension = os.path.splitext(filename)
    if file_extension.lower() not in ALLOWED_EXTENSIONS:
        res['status'] = STATUS_ERROR
        res['message'] = "{} is not supported format.".format(file_extension)
        return res
    return None


def save_file(file):
    """ Saves the file to a named temporary file """
    with NamedTemporaryFile(suffix=file_extension) as tmp_saved_audio_file:
        file.save(tmp_saved_audio_file.name)
        return tmp_saved_audio_file.name


def transcribe_audio(audio_path, spect_parser, model, decoder, device, precision):
    """ Transcribes the given audio using the deepspeech model """
    logging.info('Transcribing file...')
    transcription, _ = run_transcribe(
        audio_path=audio_path,
        spect_parser=spect_parser,
        model=model,
        decoder=decoder,
        device=device,
        precision=precision
    )
    logging.info('File transcribed')
    return transcription


@app.route('/transcribe', methods=['POST'])
def transcribe_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            res = {}
            res['status'] = STATUS_ERROR
            res['message'] = "audio file should be passed for the transcription"
            return jsonify(res)
        file = request.files['file']
        file_validation = validate_file(file)
        if file_validation is not None:
            return jsonify(file_validation)
        tmp_saved_audio_file_path = save_file(file)
        transcription = transcribe_audio(
                audio_path=tmp_saved_audio_file,
                spect_parser=spect_parser,
                model=model,
                decoder=decoder,
                device=device,
                precision=config.model.precision
            )
        res = {}
        res['status'] = STATUS_OK
        res['transcription'] = transcription
        return jsonify(res)


@hydra.main(config_name="config")
def main(cfg: ServerConfig):
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info('Setting up server...')
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )

    spect_parser = SpectrogramParser(
        audio_conf=model.spect_cfg,
        normalize=True
    )

    logging.info('Server initialised')
    app.run(
        host=cfg.host,
        port=cfg.port,
        debug=True,
        use_reloader=False
    )


if __name__ == "__main__":
    main()