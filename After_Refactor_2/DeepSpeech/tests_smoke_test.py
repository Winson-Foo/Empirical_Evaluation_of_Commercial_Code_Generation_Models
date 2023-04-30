import shutil
import tempfile
import unittest

from deepspeech import DeepSpeech, DeepSpeechConfig, EvalConfig, ModelConfig, BiDirectionalConfig, \
    AdamConfig, DataConfig, DeepSpeechTrainerConf, LMConfig, transcribe, evaluate, download_an4_dataset, \
    get_data_paths
from deepspeech import get_data_paths


class AN4SmokeTest(unittest.TestCase):
    def setUp(self):
        self.target_dir = tempfile.mkdtemp()
        self.manifest_dir = tempfile.mkdtemp()
        self.model_dir = tempfile.mkdtemp()

        self.deep_speech = DeepSpeech(self.target_dir, self.manifest_dir, self.model_dir)

    def tearDown(self):
        shutil.rmtree(self.target_dir)
        shutil.rmtree(self.manifest_dir)
        shutil.rmtree(self.model_dir)

    def test_train_eval_inference(self):
        # Hardcoded sizes to reduce memory/time, and disabled GPU
        model_cfg = BiDirectionalConfig(
            hidden_size=10,
            hidden_layers=1
        )
        train_path, val_path, test_path = self.deep_speech.download_an4(DatasetConfig())

        deep_speech_config = DeepSpeechConfig(
            trainer=DeepSpeechTrainerConf(
                max_epochs=1,
                precision=32,
                gpus=0,
                enable_checkpointing=True,
                limit_train_batches=1,
                limit_val_batches=1
            ),
            data=DataConfig(
                train_path=train_path,
                val_path=val_path,
                batch_size=10
            ),
            optim=AdamConfig(),
            model=model_cfg,
            checkpoint=ModelCheckpointConf(
                dirpath=self.model_dir,
                save_last=True,
                verbose=True
            )
        )

        self.deep_speech.train_model(deep_speech_config)

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

        print("Running Inference Smoke Tests")
        for lm_config in lm_configs:
            self.deep_speech.evaluate_model(
                EvalConfig(
                    model=ModelConfig(
                        cuda=False,
                        model_path=model_path,
                        precision=32
                    ),
                    lm=lm_config,
                    test_path=test_path
                )
            )

            # Select one file from our test manifest to run inference
            if os.path.isdir(test_path):
                file_path = next(Path(test_path).rglob('*.wav'))
            else:
                with open(test_path) as f:
                    manifest = json.load(f)
                    file_name = manifest['samples'][0]['wav_path']
                    directory = manifest['root_path']
                    file_path = os.path.join(directory, file_name)

            self.deep_speech.transcribe(
                TranscribeConfig(
                    model=ModelConfig(
                        cuda=False,
                        model_path=model_path,
                        precision=32
                    ),
                    lm=lm_config,
                    audio_path=file_path
                )
            )

    def test_train_eval_inference_folder(self):
        # Hardcoded sizes to reduce memory/time, and disabled GPU due to using TravisCI
        model_cfg = BiDirectionalConfig(
            hidden_size=10,
            hidden_layers=1
        )
        train_path, val_path, test_path = self.deep_speech.download_an4(DatasetConfig())

        deep_speech_config = DeepSpeechConfig(
            trainer=DeepSpeechTrainerConf(
                max_epochs=1,
                precision=32,
                gpus=0,
                enable_checkpointing=True,
                limit_train_batches=1,
                limit_val_batches=1
            ),
            data=DataConfig(
                train_path=train_path,
                val_path=val_path,
                batch_size=10
            ),
            optim=AdamConfig(),
            model=model_cfg,
            checkpoint=ModelCheckpointConf(
                dirpath=self.model_dir,
                save_last=True,
                verbose=True
            )
        )

        self.deep_speech.train_model(deep_speech_config)

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

        print("Running Inference Smoke Tests")
        for lm_config in lm_configs:
            self.deep_speech.evaluate_model(
                EvalConfig(
                    model=ModelConfig(
                        cuda=False,
                        model_path=model_path,
                        precision=32
                    ),
                    lm=lm_config,
                    test_path=test_path
                )
            )

            # Select one file from our test manifest to run inference
            if os.path.isdir(test_path):
                file_path = next(Path(test_path).rglob('*.wav'))
            else:
                with open(test_path) as f:
                    manifest = json.load(f)
                    file_name = manifest['samples'][0]['wav_path']
                    directory = manifest['root_path']
                    file_path = os.path.join(directory, file_name)

            self.deep_speech.transcribe(
                TranscribeConfig(
                    model=ModelConfig(
                        cuda=False,
                        model_path=model_path,
                        precision=32
                    ),
                    lm=lm_config,
                    audio_path=file_path
                )
            )


class DatasetConfig:
    target_dir: str = ''
    manifest_dir: str = ''
    min_duration: float = 0
    max_duration: float = 15
    val_fraction: float = 0.1
    sample_rate: int = 16000
    num_workers: int = 4


if __name__ == '__main__':
    unittest.main()