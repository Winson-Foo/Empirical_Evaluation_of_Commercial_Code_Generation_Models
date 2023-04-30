import logging

import hydra
import torch
from flask import Flask, request, jsonify

from deepspeech_pytorch.configs import ServerConfig
from deepspeech_pytorch.inference import transcribe_audio_file
from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.utils import load_model, load_decoder

app = Flask(__name__)

logger = logging.getLogger(__name__)


@hydra.main(config_name="config")
def main(cfg: ServerConfig):
    logger.info('Setting up server...')
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

    logger.info('Server initialised')

    @app.route('/transcribe', methods=['POST'])
    def transcribe():
        if request.method == 'POST':
            file = request.files['file']
            res = transcribe_audio_file(file, cfg.allowed_extensions, spect_parser, model, decoder, device, cfg.model.precision)
            return jsonify(res)

    app.run(
        host=cfg.host,
        port=cfg.port,
        debug=cfg.debug,
        use_reloader=False
    )


if __name__ == '__main__':
    main()