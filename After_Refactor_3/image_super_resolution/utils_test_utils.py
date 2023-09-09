import logging
import os
import unittest
from unittest.mock import patch, Mock

import yaml

from ISR.utils import utils


class TestUtilsClass(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        pass
    
    # PARAMETER CHECKING
    
    def test_check_parameter_keys(self):
        par = {'a': 0}
        utils.check_parameter_keys(parameter=par, needed_keys=['a'])
        utils.check_parameter_keys(
            parameter=par, needed_keys=None, optional_keys=['b'], default_value=-1
        )
        self.assertEqual(par['b'], -1)
        with self.assertRaises(Exception):
            utils.check_parameter_keys(parameter=par, needed_keys=['c'])
            
    # CONFIG FROM WEIGHTS
    
    def test_config_from_weights_valid(self):
        weights_path = os.path.join('a', 'path', 'to', 'rdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None}
        expected_params = {'C': 3, 'D': 1, 'G': 7, 'G0': 5, 'x': 2}
        name = 'rdn'
        utils.get_config_from_weights = Mock(return_value=expected_params)
        generated_param = utils.get_config_from_weights(
            w_path=weights_path, arch_params=arch_params, name=name
        )
        self.assertEqual(generated_param, expected_params)
    
    def test_config_from_weights_invalid(self):
        weights_path = os.path.join('a', 'path', 'to', 'rrdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None, 'T': None}
        name = 'rdn'
        with self.assertRaises(Exception):
            utils.get_config_from_weights(
                w_path=weights_path, arch_params=arch_params, name=name
            )
            
    # SETUP
    
    @patch('ISR.utils.utils.open', create=True)
    def test_setup_default_training(self, mock_file):
        mock_file.return_value.__enter__.return_value = io.StringIO("""
default:
  generator: rrdn
  feature_extractor: false
  discriminator: false
  training_set: div2k-x4
  test_set: dummy
""")
        base_conf = yaml.load(mock_file.return_value._value())
        training = True
        prediction = False
        default = True
        with patch('ISR.utils.utils.select_option', return_value='0'):
            session_type, generator, conf, dataset = utils.setup(
                'tests/data/config.yml', default, training, prediction
            )
        self.assertEqual(session_type, 'training')
        self.assertEqual(generator, 'rrdn')
        self.assertEqual(conf, base_conf)
        self.assertEqual(dataset, 'div2k-x4')
    
    @patch('ISR.utils.utils.open', create=True)
    def test_setup_default_prediction(self, mock_file):
        mock_file.return_value.__enter__.return_value = io.StringIO("""
default:
  generator: rdn
  feature_extractor: false
  discriminator: false
  training_set: div2k-x4
  test_set: dummy
generators:
  rdn:
    C: null
    D: null
    G: null
    G0: null
    x: null
weights_paths:
  generator: a/path/to/rdn-C3-D1-G7-G05-x2
""")
        base_conf = yaml.load(mock_file.return_value._value())
        training = False
        prediction = True
        default = True
        with patch('ISR.utils.utils.select_option', return_value='0'):
            session_type, generator, conf, dataset = utils.setup(
                'tests/data/config.yml', default, training, prediction
            )
        self.assertEqual(session_type, 'prediction')
        self.assertEqual(generator, 'rdn')
        self.assertEqual(conf, base_conf)
        self.assertEqual(dataset, 'dummy')
    
    # UTILITY FUNCTIONS
    
    def test__get_parser(self):
        parser = utils._get_parser()
        cl_args = parser.parse_args(['--training'])
        namespace = dict(cl_args._get_kwargs())
        self.assertEqual(namespace['training'], True)
        self.assertEqual(namespace['prediction'], False)
        self.assertEqual(namespace['default'], False)
    
    @patch('builtins.input', return_value='1')
    def test_select_option(self, input_mock):
        self.assertEqual(utils.select_option(['0', '1'], ''), '1')
        self.assertNotEqual(utils.select_option(['0', '1'], ''), '0')
    
    @patch('builtins.input', return_value='2 0')
    def test_select_multiple_options(self, input_mock):
        self.assertEqual(utils.select_multiple_options(['0', '1', '3'], ''), ['3', '0'])
        self.assertNotEqual(utils.select_multiple_options(['0', '1', '3'], ''), ['0', '3'])
    
    @patch('builtins.input', return_value='1')
    def test_select_positive_integer(self, input_mock):
        self.assertEqual(utils.select_positive_integer(''), 1)
        self.assertNotEqual(utils.select_positive_integer(''), 0)
    
    @patch('builtins.input', return_value='1.3')
    def test_select_positive_float(self, input_mock):
        self.assertEqual(utils.select_positive_float(''), 1.3)
        self.assertNotEqual(utils.select_positive_float(''), 0)
    
    @patch('builtins.input', return_value='y')
    def test_select_bool_true(self, input_mock):
        self.assertEqual(utils.select_bool(''), True)
        self.assertNotEqual(utils.select_bool(''), False)
    
    @patch('builtins.input', return_value='n')
    def test_select_bool_false(self, input_mock):
        self.assertEqual(utils.select_bool(''), False)
        self.assertNotEqual(utils.select_bool(''), True)
    
    @patch('builtins.input', return_value='0')
    def test_browse_weights(self, input_mock):
        def folder_weights_select(inp):
            if inp == '':
                return ['folder']
            if inp == 'folder':
                return ['1.hdf5']
        
        with patch('os.listdir', side_effect=folder_weights_select):
            weights = utils.browse_weights('')
        self.assertEqual(weights, 'folder/1.hdf5')
    
    def test_select_dataset(self):
        conf = {
            'test_sets': {'test_test_set': {}},
            'training_sets': {'test_train_set': {}},
        }
        with patch('ISR.utils.utils.select_option', return_value='0'):
            tr_data = utils.select_dataset('training', conf)
        with patch('ISR.utils.utils.select_option', return_value='0'):
            pr_data = utils.select_dataset('prediction', conf)
        self.assertEqual(tr_data, 'test_train_set')
        self.assertEqual(pr_data, 'test_test_set')
    
    def test_suggest_metrics(self):
        metrics = utils.suggest_metrics(
            discriminator=False, feature_extractor=False, loss_weights={}
        )
        self.assertIn('val_loss', metrics)
        self.assertNotIn('val_generator_loss', metrics)
        metrics = utils.suggest_metrics(
            discriminator=True, feature_extractor=False, loss_weights={}
        )
        self.assertIn('val_generator_loss', metrics)
        self.assertNotIn('val_feature_extractor_loss', metrics)
        self.assertNotIn('val_loss', metrics)
        metrics = utils.suggest_metrics(discriminator=True, feature_extractor=True, loss_weights={})
        self.assertIn('val_feature_extractor_loss', metrics)
        self.assertIn('val_generator_loss', metrics)
        self.assertNotIn('val_loss', metrics)
        metrics = utils.suggest_metrics(
            discriminator=False, feature_extractor=True, loss_weights={}
        )
        self.assertIn('val_feature_extractor_loss', metrics)
        self.assertIn('val_generator_loss', metrics)
        self.assertNotIn('val_loss', metrics)