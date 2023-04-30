PRETRAINED_URLS = [
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/ted_pretrained_v3.ckpt'
]

LM_URL = 'http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz'

def download_file(url, target_dir):
    file_name = os.path.basename(url)
    file_path = os.path.join(target_dir, file_name)
    if not os.path.exists(file_path):
        wget.download(url, target_dir)
    return file_path

def run_pretrained_smoke_test(target_dir, manifest_dir):
    # Disabled GPU due to using TravisCI
    cuda, precision = False, 32
    train_manifest, val_manifest, test_manifest = download_data(
        DatasetConfig(
            target_dir=target_dir,
            manifest_dir=manifest_dir
        ),
        folders=False
    )
    lm_file = download_file(LM_URL, target_dir)
    for pretrained_url in PRETRAINED_URLS:
        print("Running Pre-trained Smoke test for: ", pretrained_url)
        pretrained_file = download_file(pretrained_url, target_dir)
        pretrained_path = os.path.abspath(pretrained_file)

        lm_configs = [
            LMConfig(),  # Greedy
            LMConfig(
                decoder_type=DecoderType.beam
            ),  # Test Beam Decoder
            LMConfig(
                decoder_type=DecoderType.beam,
                lm_path=os.path.basename(lm_file),
                alpha=1,
                beta=1
            )  # Test Beam Decoder with LM
        ]

        for lm_config in lm_configs:
            eval_model(
                model_path=pretrained_path,
                test_path=test_manifest,
                cuda=cuda,
                lm_config=lm_config,
                precision=precision
            )
            inference(
                test_path=test_manifest,
                model_path=pretrained_path,
                cuda=cuda,
                lm_config=lm_config,
                precision=precision
            )

import os
import wget

from deepspeech_pytorch.configs.inference_config import LMConfig
from deepspeech_pytorch.enums import DecoderType
from tests.smoke_test import DatasetConfig, DeepSpeechSmokeTest

PRETRAINED_URLS = [
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/an4_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/librispeech_pretrained_v3.ckpt',
    'https://github.com/SeanNaren/deepspeech.pytorch/releases/download/V3.0/ted_pretrained_v3.ckpt'
]

LM_URL = 'http://www.openslr.org/resources/11/3-gram.pruned.3e-7.arpa.gz'


def download_file(url, target_dir):
    file_name = os.path.basename(url)
    file_path = os.path.join(target_dir, file_name)
    if not os.path.exists(file_path):
        wget.download(url, target_dir)
    return file_path


def run_pretrained_smoke_test(target_dir, manifest_dir):
    # Disabled GPU due to using TravisCI
    cuda, precision = False, 32
    train_manifest, val_manifest, test_manifest = download_data(
        DatasetConfig(
            target_dir=target_dir,
            manifest_dir=manifest_dir
        ),
        folders=False
    )
    lm_file = download_file(LM_URL, target_dir)
    for pretrained_url in PRETRAINED_URLS:
        print("Running Pre-trained Smoke test for: ", pretrained_url)
        pretrained_file = download_file(pretrained_url, target_dir)
        pretrained_path = os.path.abspath(pretrained_file)

        lm_configs = [
            LMConfig(),  # Greedy
            LMConfig(
                decoder_type=DecoderType.beam
            ),  # Test Beam Decoder
            LMConfig(
                decoder_type=DecoderType.beam,
                lm_path=os.path.basename(lm_file),
                alpha=1,
                beta=1
            )  # Test Beam Decoder with LM
        ]

        for lm_config in lm_configs:
            eval_model(
                model_path=pretrained_path,
                test_path=test_manifest,
                cuda=cuda,
                lm_config=lm_config,
                precision=precision
            )
            inference(
                test_path=test_manifest,
                model_path=pretrained_path,
                cuda=cuda,
                lm_config=lm_config,
                precision=precision
            )


if __name__ == '__main__':
    target_dir = 'data'
    manifest_dir = 'manifest'
    run_pretrained_smoke_test(target_dir, manifest_dir)