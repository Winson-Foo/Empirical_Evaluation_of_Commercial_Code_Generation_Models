import unittest
from unittest.mock import patch

from ISR import assistant


class TestISRAssistant(unittest.TestCase):
    def setUp(self):
        self.default_config = {
            'default': {
                'feature_extractor': False,
                'discriminator': False,
                'generator': 'rdn',
                'training_set': 'test',
                'test_set': 'test',
            },
            'session': {
                'training': {
                    'patch_size': 0,
                    'epochs': 0,
                    'steps_per_epoch': 0,
                    'batch_size': 0,
                    'n_validation_samples': None,
                    'metrics': None,
                    'learning_rate': {},
                    'adam_optimizer': None,
                    'flatness': None,
                    'fallback_save_every_n_epochs': None,
                    'monitored_metrics': None
                },
                'prediction': {
                    'patch_size': 5
                }
            },
            'generators': {
                'rdn': {
                    'x': 0
                }
            },
            'training_sets': {
                'test': {
                    'lr_train_dir': None,
                    'hr_train_dir': None,
                    'lr_valid_dir': None,
                    'hr_valid_dir': None,
                    'data_name': None
                }
            },
            'loss_weights': None,
            'losses': None,
            'log_dirs': {
                'logs': None,
                'weights': None
            },
            'weights_paths': {
                'generator': 'a/path/rdn-C1-D6-G1-G02-x0-weights.hdf5',
                'discriminator': 'a/path/rdn-weights.hdf5'
            }
        }

    def tearDown(self):
        pass

    @patch('ISR.assistant._get_module', return_value=Object())
    @patch('ISR.train.trainer.Trainer', return_value=Object())
    def test_run_creates_trainer_with_correct_arguments(self, trainer_mock, _get_module_mock):
        with patch('yaml.load', return_value=self.default_config):
            assistant.run(
                config_file='tests/data/config.yml', training=True, prediction=False, default=True
            )
            trainer_mock.assert_called_once()

    @patch('ISR.assistant._get_module', return_value=Object())
    @patch('ISR.predict.predictor.Predictor', return_value=Object())
    def test_run_creates_predictor_with_correct_arguments(self, predictor_mock, _get_module_mock):
        with patch('yaml.load', return_value=self.default_config):
            assistant.run(
                config_file='tests/data/config.yml', training=False, prediction=True, default=True
            )
            predictor_mock.assert_called_once()


class Object:
    def __init__(self, *args, **kwargs):
        self.scale = 0
        self.patch_size = 0
        pass

    def make_model(self, *args, **kwargs):
        return self

    def train(self, *args, **kwargs):
        return True

    def get_predictions(self, *args, **kwargs):
        return True