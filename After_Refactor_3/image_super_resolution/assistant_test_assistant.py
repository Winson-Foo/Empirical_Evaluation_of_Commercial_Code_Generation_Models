import logging
import os
import unittest
from unittest.mock import patch

from ISR import assistant


class MockObject:
    def __init__(self, *args, **kwargs):
        self.scale = 0
        self.patch_size = 0
    
    def make_model(self, *args, **kwargs):
        return self
    
    def train(self, *args, **kwargs):
        return True
    
    def get_predictions(self, *args, **kwargs):
        return True


class TestAssistant(unittest.TestCase):
    CONFIG_FILE = os.path.join('tests', 'data', 'config.yml')
    DEFAULT_CONFIG = {
        'generator': 'rdn',
        'training_set': 'test',
        'test_set': 'test',
        'prediction': {'patch_size': 5},
    }
    
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
    
    @patch('ISR.assistant._get_module', return_value=MockObject())
    @patch('ISR.train.trainer.Trainer', return_value=MockObject())
    def test_run_training(self, trainer, _get_module):
        config = self._load_config()
        assistant.run(config_file=self.CONFIG_FILE, training=True, prediction=False, default=True)
        trainer.assert_called_once()
    
    @patch('ISR.assistant._get_module', return_value=MockObject())
    @patch('ISR.predict.predictor.Predictor', return_value=MockObject())
    def test_run_prediction(self, predictor, _get_module):
        config = self._load_config()
        assistant.run(config_file=self.CONFIG_FILE, training=False, prediction=True, default=True)
        predictor.assert_called_once()
    
    def _load_config(self):
        config = self.DEFAULT_CONFIG.copy()
        loaded_config = yaml.load(open(self.CONFIG_FILE, 'r'))
        config.update(loaded_config.get('default', {}))
        return config


if __name__ == '__main__':
    unittest.main()