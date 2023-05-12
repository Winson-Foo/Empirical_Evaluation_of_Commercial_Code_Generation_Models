import logging.config
import os
import unittest
from unittest.mock import patch

from ISR import assistant
from ISR import Object

logging.config.fileConfig(os.path.join("tests", "config_files", "logging.conf"))
log = logging.getLogger(__name__)

class TestAssistant(unittest.TestCase):
    
    def setUp(self):
        self.config_file = os.path.join("tests", "config_files", "config.yml")
        assistant.conf = {}
        assistant.load_config(self.config_file)
    
    def tearDown(self):
        pass
    
    @patch('ISR.assistant._get_module', return_value=Object())
    @patch('ISR.train.trainer.Trainer', return_value=Object())
    def test_run_trainer(self, trainer, _get_module):
        assistant.run(config_file=self.config_file, training=True, prediction=False, default=True)
        self.assertTrue(trainer.called)
    
    @patch('ISR.assistant._get_module', return_value=Object())
    @patch('ISR.predict.predictor.Predictor', return_value=Object())
    def test_run_predictor(self, predictor, _get_module):
        assistant.run(config_file=self.config_file, training=False, prediction=True, default=True)
        self.assertTrue(predictor.called)