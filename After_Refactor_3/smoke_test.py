import os
import shutil
import tempfile
import unittest

from deepspeech_pytorch.configs.model_config import BiDirectionalConfig
from deepspeech_pytorch.training.loader import download_data
from deepspeech_pytorch.training.trainer import train_model, evaluate_model, transcribe_audio
from configs import DatasetConfig, DeepSpeechConfig

DATASET_CONFIG = DatasetConfig(
    target_dir=tempfile.mkdtemp(),
    manifest_dir=tempfile.mkdtemp()
)
TRAIN_CONFIG = DeepSpeechConfig(
    limit_train_batches=1,
    limit_val_batches=1,
    max_epochs=1,
    batch_size=10,
    precision=32,
    gpus=0
)
MODEL_CONFIG = BiDirectionalConfig(
    hidden_size=10,
    hidden_layers=1
)


class DeepSpeechSmokeTest(unittest.TestCase):
    def setUp(self):
        os.makedirs(DATASET_CONFIG.target_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(DATASET_CONFIG.target_dir)
        shutil.rmtree(DATASET_CONFIG.manifest_dir)

    def build_train_evaluate_model(self, folders=False):
        train_path, val_path, test_path = download_data(DATASET_CONFIG, folders=folders)
        model = train_model(TRAIN_CONFIG, train_path, val_path, DATASET_CONFIG.target_dir)
        evaluate_model(TRAIN_CONFIG, test_path, model)
        transcribe_audio(TRAIN_CONFIG, test_path, model)

    def test_train_eval_inference(self):
        self.build_train_evaluate_model(folders=False)

    def test_train_eval_inference_folder(self):
        self.build_train_evaluate_model(folders=True)


if __name__ == '__main__':
    unittest.main()