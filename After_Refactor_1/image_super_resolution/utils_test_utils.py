import logging
import os
import unittest
import yaml
from unittest.mock import patch, MagicMock
from ISR.utils import utils

class UtilsClassTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
    
    def setUp(self):
        self.utils = utils
    
    def test_check_parameter_keys_with_no_optional_keys(self):
        par = {'a': 0}
        self.utils.check_parameter_keys(parameter=par, needed_keys=['a'])
        self.assertRaises(KeyError, self.utils.check_parameter_keys, parameter=par, needed_keys=['b'])
    
    def test_check_parameter_keys_with_optional_keys(self):
        par = {'a': 0}
        self.utils.check_parameter_keys(parameter=par, needed_keys=['a'], optional_keys=['b'], default_value=-1)
        self.assertEqual(par['b'], -1)
    
    def test_config_from_weights_with_valid_weights(self):
        weights = os.path.join('a', 'path', 'to', 'rdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None}
        expected_params = {'C': 3, 'D': 1, 'G': 7, 'G0': 5, 'x': 2}
        name = 'rdn'
        generated_param = self.utils.get_config_from_weights(w_path=weights, arch_params=arch_params, name=name)
        for p in expected_params:
            self.assertTrue(generated_param[p] == expected_params[p])
    
    def test_config_from_weights_with_invalid_weights(self):
        weights = os.path.join('a', 'path', 'to', 'rrdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None, 'T': None}
        name = 'rdn'
        self.assertRaises(ValueError, self.utils.get_config_from_weights, w_path=weights, arch_params=arch_params, name=name)
    
    @patch('yaml.load', MagicMock(return_value={'default': {
            'generator': 'rrdn',
            'feature_extractor': False,
            'discriminator': False,
            'training_set': 'div2k-x4',
            'test_set': 'dummy',
        }}))
    def test_setup_default_training(self):
        base_conf = {}
        training = True
        prediction = False
        default = True
        
        session_type, generator, conf, dataset = self.utils.setup('tests/data/config.yml', default, training, prediction)
        self.assertTrue(session_type == 'training')
        self.assertTrue(generator == 'rrdn')
        self.assertTrue(conf == base_conf)
        self.assertTrue(dataset == 'div2k-x4')
    
    @patch('yaml.load', MagicMock(return_value={'default': {
            'generator': 'rdn',
            'feature_extractor': False,
            'discriminator': False,
            'training_set': 'div2k-x4',
            'test_set': 'dummy',
        },
        'generators': {'rdn': {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None}},
        'weights_paths': {'generator': os.path.join('a', 'path', 'to', 'rdn-C3-D1-G7-G05-x2')}}))
    def test_setup_default_prediction(self):
        base_conf = {}
        training = False
        prediction = True
        default = True
        
        session_type, generator, conf, dataset = self.utils.setup(
            'tests/data/config.yml', default, training, prediction
        )
        self.assertTrue(session_type == 'prediction')
        self.assertTrue(generator == 'rdn')
        self.assertTrue(conf == base_conf)
        self.assertTrue(dataset == 'dummy')
    
    def test__get_parser(self):
        parser = self.utils._get_parser()
        cl_args = parser.parse_args(['--training'])
        namespace = cl_args._get_kwargs()
        self.assertTrue(('training', True) in namespace)
        self.assertTrue(('prediction', False) in namespace)
        self.assertTrue(('default', False) in namespace)
    
    @patch('builtins.input', MagicMock(return_value='1'))
    def test_select_option(self):
        self.assertEqual(self.utils.select_option(['0', '1'], ''), '1')
        self.assertNotEqual(self.utils.select_option(['0', '1'], ''), '0')
    
    @patch('builtins.input', MagicMock(return_value='2 0'))
    def test_select_multiple_options(self):
        self.assertEqual(self.utils.select_multiple_options(['0', '1', '3'], ''), ['3', '0'])
        self.assertNotEqual(self.utils.select_multiple_options(['0', '1', '3'], ''), ['0', '3'])
    
    @patch('builtins.input', MagicMock(return_value='1'))
    def test_select_positive_integer(self):
        self.assertEqual(self.utils.select_positive_integer(''), 1)
        self.assertNotEqual(self.utils.select_positive_integer(''), 0)
    
    @patch('builtins.input', MagicMock(return_value='1.3'))
    def test_select_positive_float(self):
        self.assertEqual(self.utils.select_positive_float(''), 1.3)
        self.assertNotEqual(self.utils.select_positive_float(''), 0)
    
    @patch('builtins.input', MagicMock(return_value='y'))
    def test_select_bool_true(self):
        self.assertEqual(self.utils.select_bool(''), True)
        self.assertNotEqual(self.utils.select_bool(''), False)
    
    @patch('builtins.input', MagicMock(return_value='n'))
    def test_select_bool_false(self):
        self.assertEqual(self.utils.select_bool(''), False)
        self.assertNotEqual(self.utils.select_bool(''), True)
    
    @patch('builtins.input', MagicMock(return_value='0'))
    def test_browse_weights(self):
        def folder_weights_select(inp):
            if inp == '':
                return ['folder']
            if inp == 'folder':
                return ['1.hdf5']
        
        with patch('os.listdir', side_effect=folder_weights_select):
            weights = self.utils.browse_weights('')
        self.assertEqual(weights, 'folder/1.hdf5')
    
    def test_select_dataset(self):
        conf = yaml.load(open(os.path.join('tests', 'data', 'config.yml'), 'r'))
        conf['test_sets'] = {'test_test_set': {}}
        conf['training_sets'] = {'test_train_set': {}}
        
        with patch('builtins.input', MagicMock(return_value='1')):
            tr_data = self.utils.select_dataset('training', conf)
            
        with patch('builtins.input', MagicMock(return_value='1')):
            pr_data = self.utils.select_dataset('prediction', conf)
            
        self.assertEqual(tr_data, 'test_train_set')
        self.assertEqual(pr_data, 'test_test_set')
    
    def test_suggest_metrics_when_discriminator_is_false(self):
        metrics = self.utils.suggest_metrics(discriminator=False, feature_extractor=False, loss_weights={})
        self.assertTrue('val_loss' in metrics)
        self.assertFalse('val_generator_loss' in metrics)
    
    def test_suggest_metrics_when_discriminator_is_true_and_feature_extractor_is_false(self):
        metrics = self.utils.suggest_metrics(discriminator=True, feature_extractor=False, loss_weights={})
        self.assertTrue('val_generator_loss' in metrics)
        self.assertFalse('val_feature_extractor_loss' in metrics)
        self.assertFalse('val_loss' in metrics)
    
    def test_suggest_metrics_when_discriminator_and_feature_extractor_both_are_true(self):
        metrics = self.utils.suggest_metrics(discriminator=True, feature_extractor=True, loss_weights={})
        self.assertTrue('val_feature_extractor_loss' in metrics)
        self.assertTrue('val_generator_loss' in metrics)
        self.assertFalse('val_loss' in metrics)
    
    def test_suggest_metrics_when_feature_extractor_is_true(self):
        metrics = self.utils.suggest_metrics(discriminator=False, feature_extractor=True, loss_weights={})
        self.assertTrue('val_feature_extractor_loss' in metrics)
        self.assertTrue('val_generator_loss' in metrics)
        self.assertFalse('val_loss' in metrics)