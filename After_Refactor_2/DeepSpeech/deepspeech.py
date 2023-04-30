import json
import os
import shutil
import tempfile
from pathlib import Path

from pytorch_lightning.utilities import _module_available

from deepspeech_pytorch.configs.inference_config import EvalConfig, ModelConfig, TranscribeConfig, LMConfig
from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, AdamConfig, BiDirectionalConfig, \
    DataConfig, DeepSpeechTrainerConf
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.inference import transcribe
from deepspeech_pytorch.testing import evaluate
from deepspeech_pytorch.training import train
from deepspeech_pytorch.utils import download_an4_dataset, get_data_paths


class DeepSpeech:
    def __init__(self, target_dir, manifest_dir, model_dir):
        self.target_dir = target_dir
        self.manifest_dir = manifest_dir
        self.model_dir = model_dir

    def train_model(self, deep_speech_config):
        print("Running Training DeepSpeech Model")
        train(deep_speech_config)

    def evaluate_model(self, eval_config):
        evaluate(eval_config)

    def transcribe(self, transcribe_cfg):
        transcribe(transcribe_cfg)

    def download_an4(self, dataset_config):
        download_an4_dataset(
            target_dir=dataset_config.target_dir,
            manifest_dir=dataset_config.manifest_dir,
            min_duration=dataset_config.min_duration,
            max_duration=dataset_config.max_duration,
            num_workers=dataset_config.num_workers
        )

        # Expected output paths
        return get_data_paths(self.target_dir)