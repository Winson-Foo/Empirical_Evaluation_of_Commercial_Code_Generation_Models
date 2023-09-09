import logging
import os
import unittest
from unittest.mock import patch

import yaml
from ISR.utils import utils


logger = utils.get_logger(__name__)


def check_parameter_keys(parameter, needed_keys=None, optional_keys=None, default_value=None):
    if needed_keys:
        for key in needed_keys:
            if key not in parameter:
                logger.error('{p} is missing key {k}'.format(p=parameter, k=key))
                raise
    if optional_keys:
        for key in optional_keys:
            if key not in parameter:
                logger.info(
                    'Setting {k} in {p} to {d}'.format(k=key, p=parameter, d=default_value)
                )
                parameter[key] = default_value


class UtilsClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

    def test_check_parameter_keys(self):
        parameters = {'a': 0}
        utils.check_parameter_keys(parameter=parameters, needed_keys=['a'])
        utils.check_parameter_keys(
            parameter=parameters, optional_keys=['b'], default_value=-1
        )
        self.assertEqual(parameters['b'], -1)
        self.assertRaises(Exception, utils.check_parameter_keys, **{'parameter': parameters, 'needed_keys': ['c']})

    def test_get_config_from_weights_C(self):
        weights_path = os.path.join('a', 'path', 'to', 'rdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None}
        expected_C = 3
        name = 'rdn'

        generated_params = utils.get_config_from_weights(w_path=weights_path, arch_params=arch_params, name=name)
        self.assertEqual(generated_params['C'], expected_C)

    def test_get_config_from_weights_D(self):
        weights_path = os.path.join('a', 'path', 'to', 'rdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None}
        expected_D = 1
        name = 'rdn'

        generated_params = utils.get_config_from_weights(w_path=weights_path, arch_params=arch_params, name=name)
        self.assertEqual(generated_params['D'], expected_D)

    def test_get_config_from_weights_G(self):
        weights_path = os.path.join('a', 'path', 'to', 'rdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None}
        expected_G = 7
        name = 'rdn'

        generated_params = utils.get_config_from_weights(w_path=weights_path, arch_params=arch_params, name=name)
        self.assertEqual(generated_params['G'], expected_G)

    def test_get_config_from_weights_G0(self):
        weights_path = os.path.join('a', 'path', 'to', 'rdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None}
        expected_G0 = 5
        name = 'rdn'

        generated_params = utils.get_config_from_weights(w_path=weights_path, arch_params=arch_params, name=name)
        self.assertEqual(generated_params['G0'], expected_G0)

    def test_get_config_from_weights_x(self):
        weights_path = os.path.join('a', 'path', 'to', 'rdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None}
        expected_x = 2
        name = 'rdn'

        generated_params = utils.get_config_from_weights(w_path=weights_path, arch_params=arch_params, name=name)
        self.assertEqual(generated_params['x'], expected_x)

    def test_config_from_weights_invalid(self):
        weights_path = os.path.join('a', 'path', 'to', 'rrdn-C3-D1-G7-G05-x2')
        arch_params = {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None, 'T': None}
        name = 'rdn'
        self.assertRaises(Exception, utils.get_config_from_weights, **{'w_path': weights_path, 'arch_params': arch_params, 'name': name})

    def test_setup_default_training(self):
        base_conf = {}
        base_conf['default'] = {
            'generator': 'rrdn',
            'feature_extractor': False,
            'discriminator': False,
            'training_set': 'div2k-x4',
            'test_set': 'dummy',
        }
        training = True
        prediction = False
        default = True

        with patch('yaml.load', return_value=base_conf) as import_module:
            session_type, generator, conf, dataset = utils.setup('tests/data/config.yml', default, training, prediction)

        self.assertEqual(session_type, 'training')
        self.assertEqual(generator, 'rrdn')
        self.assertEqual(conf, base_conf)
        self.assertEqual(dataset, 'div2k-x4')

    def test_setup_default_prediction(self):
        base_conf = {}
        base_conf['default'] = {
            'generator': 'rdn',
            'feature_extractor': False,
            'discriminator': False,
            'training_set': 'div2k-x4',
            'test_set': 'dummy',
        }
        base_conf['generators'] = {'rdn': {'C': None, 'D': None, 'G': None, 'G0': None, 'x': None}}
        base_conf['weights_paths'] = {
            'generator': os.path.join('a', 'path', 'to', 'rdn-C3-D1-G7-G05-x2')
        }
        training = False
        prediction = True
        default = True

        with patch('yaml.load', return_value=base_conf):
            session_type, generator, conf, dataset = utils.setup('tests/data/config.yml', default, training, prediction)

        self.assertEqual(session_type, 'prediction')
        self.assertEqual(generator, 'rdn')
        self.assertEqual(conf, base_conf)
        self.assertEqual(dataset, 'dummy')

    def test_get_parser(self):
        parser = utils._get_parser()
        cl_args = parser.parse_args(['--training'])
        namespace = cl_args._get_kwargs()
        self.assertIn(('training', True), namespace)
        self.assertIn(('prediction', False), namespace)
        self.assertIn(('default', False), namespace)

    @patch('builtins.input', return_value='1')
    def test_select_option_single(self, input):
        self.assertEqual(utils.select_option(['0', '1'], ''), '1')
        self.assertNotEqual(utils.select_option(['0', '1'], ''), '0')

    @patch('builtins.input', return_value='2 0')
    def test_select_option_multiple(self, input):
        self.assertEqual(utils.select_multiple_options(['0', '1', '3'], ''), ['3', '0'])
        self.assertNotEqual(utils.select_multiple_options(['0', '1', '3'], ''), ['0', '3'])

    @patch('builtins.input', return_value='1')
    def test_select_positive_integer(self, input):
        self.assertEqual(utils.select_positive_integer(''), 1)
        self.assertNotEqual(utils.select_positive_integer(''), 0)

    @patch('builtins.input', return_value='1.3')
    def test_select_positive_float(self, input):
        self.assertEqual(utils.select_positive_float(''), 1.3)
        self.assertNotEqual(utils.select_positive_float(''), 0)

    @patch('builtins.input', return_value='y')
    def test_select_bool_true(self, input):
        self.assertEqual(utils.select_bool(''), True)
        self.assertNotEqual(utils.select_bool(''), False)

    @patch('builtins.input', return_value='n')
    def test_select_bool_false(self, input):
        self.assertEqual(utils.select_bool(''), False)
        self.assertNotEqual(utils.select_bool(''), True)

    @patch('builtins.input', return_value='0')
    def test_browse_weights(self, input):
        def folder_weights_select(folder):
            if folder == '':
                return ['folder']
            if folder == 'folder':
                return ['1.hdf5']

        with patch('os.listdir', side_effect=folder_weights_select):
            weights_path = utils.browse_weights('')
        self.assertEqual(weights_path, 'folder/1.hdf5')

    def test_select_dataset(self):
        configuration = yaml.load(open(os.path.join('tests', 'data', 'config.yml'), 'r'))
        configuration['test_sets'] = {'test_test_set': {}}
        configuration['training_sets'] = {'test_train_set': {}}

        training_set = utils.select_dataset('training', configuration)
        test_set = utils.select_dataset('prediction', configuration)

        self.assertEqual(training_set, 'test_train_set')
        self.assertEqual(test_set, 'test_test_set')

    def test_suggest_metrics_with_discriminator(self):
        metrics = utils.suggest_metrics(
            discriminator=True, feature_extractor=False, loss_weights={}
        )
        self.assertIn('val_generator_loss', metrics)
