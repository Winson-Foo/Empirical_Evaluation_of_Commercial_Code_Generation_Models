import json
import os
from pathlib import Path

from data.an4 import download_an4
from deepspeech_pytorch.enums import DecoderType


def download_data(cfg):
    download_an4(
        target_dir=cfg.target_dir,
        manifest_dir=cfg.manifest_dir,
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
        num_workers=cfg.num_workers
    )

    train_path = os.path.join(cfg.target_dir, 'train/')
    val_path = os.path.join(cfg.target_dir, 'val/')
    test_path = os.path.join(cfg.target_dir, 'test/')

    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)

    return train_path, val_path, test_path


def select_test_file(test_path):
    if os.path.isdir(test_path):
        file_path = next(Path(test_path).rglob('*.wav'))
    else:
        with open(test_path) as f:
            # select a file to use for inference test
            manifest = json.load(f)
            file_name = manifest['samples'][0]['wav_path']
            directory = manifest['root_path']
            file_path = os.path.join(directory, file_name)
    return file_path


def create_lm_configs():
    lm_configs = [{'lm_alpha': 0.75, 'lm_beta': 1.85}]
    if _module_available('ctcdecode'):
        lm_configs.append(
            {'lm_alpha': 1.4, 'lm_beta': 5.0, 'decoder_type': DecoderType.beam}
        )
    return lm_configs