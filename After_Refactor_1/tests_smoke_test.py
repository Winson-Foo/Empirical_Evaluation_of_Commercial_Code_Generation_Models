import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from pytorch_lightning.utilities import _module_available

from data import download_an4_data, DatasetConfig
from deepspeech_pytorch.configs.inference_config import EvalConfig, ModelConfig, TranscribeConfig, LMConfig
from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig, AdamConfig, BiDirectionalConfig, \
    DataConfig, DeepSpeechTrainerConf
from deepspeech_pytorch.enums import DecoderType
from deepspeech_pytorch.inference import transcribe
from deepspeech_pytorch.testing import evaluate
from deepspeech_pytorch.training import train

class DeepSpeechSmokeTest(unittest.TestCase):
    def setUp(self):
        self.target_dir = tempfile.mkdtemp()
        self.manifest_dir = tempfile.mkdtemp()
        self.model_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.target_dir)
        shutil.rmtree(self.manifest_dir)
        shutil.rmtree(self.model_dir)

    def build_train_evaluate_model(self,
                                   limit_train_batches: int,
                                   limit_val_batches: int,
                                   epoch: int,
                                   batch_size: int,
                                   model_config: BiDirectionalConfig,
                                   precision: int,
                                   gpus: int,
                                   folders: bool):
        cuda = gpus > 0

        train_path, val_path, test_path = download_an4_data(
            DatasetConfig(
                target_dir=self.target_dir,
                manifest_dir=self.manifest_dir
            ),
            folders=folders
        )

        train_cfg = DeepSpeechConfig(
            trainer=DeepSpeechTrainerConf(
                max_epochs=epoch,
                precision=precision,
                gpus=gpus,
                enable_checkpointing=True,
                limit_train_batches=limit_train_batches,
                limit_val_batches=limit_val_batches
            ),
            data=DataConfig(
                train_path=train_path,
                val_path=val_path,
                batch_size=batch_size
            ),
            optim=AdamConfig(),
            model=model_config,
            checkpoint=ModelCheckpointConf(
                dirpath=self.model_dir,
                save_last=True,
                verbose=True
            )
        )
        train(train_cfg)

        # Expected final model path after training
        model_path = os.path.join(self.model_dir, 'last.ckpt')
        assert os.path.exists(model_path)

        lm_configs = [LMConfig()]

        if _module_available('ctcdecode'):
            lm_configs.append(
                LMConfig(
                    decoder_type=DecoderType.beam
                )
            )

        for lm_config in lm_configs:
            self.eval_model(
                model_path=model_path,
                test_path=test_path,
                cuda=cuda,
                precision=precision,
                lm_config=lm_config
            )

            self.inference(test_path=test_path,
                           model_path=model_path,
                           cuda=cuda,
                           precision=precision,
                           lm_config=lm_config)

    def eval_model(self,
                   model_path: str,
                   test_path: str,
                   cuda: bool,
                   precision: int,
                   lm_config: LMConfig):
        # Due to using TravisCI with no GPU support we have to disable cuda
        eval_cfg = EvalConfig(
            model=ModelConfig(
                cuda=cuda,
                model_path=model_path,
                precision=precision
            ),
            lm=lm_config,
            test_path=test_path
        )
        evaluate(eval_cfg)

    def inference(self,
                  test_path: str,
                  model_path: str,
                  cuda: bool,
                  precision: int,
                  lm_config: LMConfig):
        if os.path.isdir(test_path):
            file_path = next(Path(test_path).rglob('*.wav'))
        else:
            with open(test_path) as f:
                manifest = json.load(f)
                file_name = manifest['samples'][0]['wav_path']
                directory = manifest['root_path']
                file_path = os.path.join(directory, file_name)

        transcribe_cfg = TranscribeConfig(
            model=ModelConfig(
                cuda=cuda,
                model_path=model_path,
                precision=precision
            ),
            lm=lm_config,
            audio_path=file_path
        )
        transcribe(transcribe_cfg)

class AN4SmokeTest(DeepSpeechSmokeTest):

    def test_train_eval_inference(self):
        model_cfg = BiDirectionalConfig(
            hidden_size=10,
            hidden_layers=1
        )
        self.build_train_evaluate_model(
            limit_train_batches=1,
            limit_val_batches=1,
            epoch=1,
            batch_size=10,
            model_config=model_cfg,
            precision=32,
            gpus=0,
            folders=False
        )

    def test_train_eval_inference_folder(self):
        """Test train/eval/inference using folder directories rather than manifest files"""
        model_cfg = BiDirectionalConfig(
            hidden_size=10,
            hidden_layers=1
        )
        self.build_train_evaluate_model(
            limit_train_batches=1,
            limit_val_batches=1,
            epoch=1,
            batch_size=10,
            model_config=model_cfg,
            precision=32,
            gpus=0,
            folders=True
        )

if __name__ == '__main__':
    unittest.main()