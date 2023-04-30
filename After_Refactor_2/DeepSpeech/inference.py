import logging
import os
from tempfile import NamedTemporaryFile

import torch

from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.utils import load_model, load_decoder


def run_transcribe(audio_path, spect_parser, model, decoder, device, precision):
    with torch.no_grad():
        spect = spect_parser.parse_audio(audio_path)
        if precision == 16:
            spect = spect.half()
        spect = spect.to(device)

        out, output_sizes = model(spect)
        decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
        transcription = decoded_output[0]
    return transcription, decoded_offsets


def transcribe_audio_file(audio_path, allowed_extensions, spect_parser, model, decoder, device, precision):
    res = {}
    _, file_extension = os.path.splitext(audio_path)
    if file_extension.lower() not in allowed_extensions:
        res['status'] = "error"
        res['message'] = "{} is not supported format.".format(file_extension)
    else:
        with NamedTemporaryFile(suffix=file_extension) as tmp_saved_audio_file:
            tmp_saved_audio_file.write(open(audio_path, "rb").read())
            logging.info('Transcribing audio file...')
            transcription, _ = run_transcribe(
                audio_path=tmp_saved_audio_file.name,
                spect_parser=spect_parser,
                model=model,
                decoder=decoder,
                device=device,
                precision=precision
            )
            logging.info('File audio transcribed')
            res['status'] = "OK"
            res['transcription'] = transcription
    return res