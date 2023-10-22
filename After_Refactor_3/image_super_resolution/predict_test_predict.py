import logging
import unittest
import shutil
from copy import copy

import yaml
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock

from ISR.models.rdn import RDN
from ISR.predict.predictor import Predictor


class PredictorClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls.setup = yaml.load(Path('tests/data/config.yml').read_text(), Loader=yaml.FullLoader)
        cls.RDN = RDN(arch_params=cls.setup['rdn'], patch_size=cls.setup['patch_size'])
        
        cls.temp_data = Path('tests/temporary_test_data')
        cls.temp_data.mkdir(parents=True, exist_ok=True)
        cls.valid_files = cls.temp_data / 'valid_files'
        cls.valid_files.mkdir()
        cls.invalid_files = cls.temp_data / 'invalid_files'
        cls.invalid_files.mkdir()
        cls.out_dir = cls.temp_data / 'out_dir'
        cls.out_dir.mkdir()
        
        for item in ['data2.gif', 'data1.png', 'data0.jpeg']:
            (cls.valid_files / item).touch()
        
        for item in ['data2.gif', 'data.data', 'data02']:
            (cls.invalid_files / item).touch()
        
        def nullifier(*args):
            pass
        
        cls.predictor = Predictor(input_dir=str(cls.valid_files), output_dir=str(cls.out_dir))
        cls.predictor.logger = Mock(return_value=True)
    
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_data)
    
    def setUp(self):
        self.pred = copy(self.predictor)
    
    def tearDown(self):
        pass
    
    def test_load_weights_with_no_weights_raises_exception(self):
        self.pred.weights_path = None

        with self.assertRaises(Exception):
            self.pred._load_weights()

    def test_load_weights_with_valid_weights_raises_value_error(self):
        def raise_path(path):
            raise ValueError(path)

        self.pred.model = self.RDN
        self.pred.model.model.load_weights = Mock(side_effect=raise_path)
        self.pred.weights_path = 'a/path'

        with self.assertRaises(ValueError):
            self.pred._load_weights()

    def test_make_basename_returns_expected_value(self):
        self.pred.model = self.RDN
        made_name = self.pred._make_basename()
        self.assertEqual(made_name, 'rdn-C3-D10-G64-G064-x2')

    def test_forward_pass_pixel_range_and_type(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr

        self.pred.model = self.RDN
        self.pred.model.model.predict = Mock(side_effect=valid_sr_output)

        with patch('imageio.imread', return_value=np.random.random((10, 10, 3))):
            sr = self.pred._forward_pass('file_path')

        self.assertTrue(type(sr[0, 0, 0]) is np.uint8)
        self.assertTrue(np.all(sr >= 0.0))
        self.assertTrue(np.all(sr <= 255.0))
        self.assertTrue(np.any(sr > 1.0))
        self.assertTrue(sr.shape == (20, 20, 3))

    def test_forward_pass_with_4_channels_returns_none(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr

        self.pred.model = self.RDN
        self.pred.model.model.predict = Mock(side_effect=valid_sr_output)

        with patch('imageio.imread', return_value=np.random.random((10, 10, 4))):
            sr = self.pred._forward_pass('file_path')

        self.assertIsNone(sr)

    def test_forward_pass_with_1_channel_returns_none(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr

        self.pred.model = self.RDN
        self.pred.model.model.predict = Mock(side_effect=valid_sr_output)

        with patch('imageio.imread', return_value=np.random.random((10, 10, 1))):
            sr = self.pred._forward_pass('file_path')

        self.assertIsNone(sr)

    def test_get_predictions_calls_load_weights_and_forward_pass(self):
        self.pred._load_weights = Mock(return_value={})
        self.pred._forward_pass = Mock(return_value=True)

        with patch('imageio.imwrite', return_value=True):
            self.pred.get_predictions(self.RDN, 'a/path/arch-weights_session1_session2.hdf5')

        self.pred._load_weights.assert_called_once()
        self.pred._forward_pass.assert_called()

    def test_output_folder_and_dataname_are_set_correctly(self):
        self.assertEqual(self.pred.data_name, 'valid_files')
        self.assertEqual(self.pred.output_dir, Path('tests/temporary_test_data/out_dir/valid_files'))

    def test_valid_extensions_returns_expected_value(self):
        predictor = Predictor(input_dir=str(self.valid_files), output_dir=str(self.out_dir))

        self.assertListEqual(sorted(predictor.img_ls), sorted([self.valid_files / 'data0.jpeg', self.valid_files / 'data1.png']))

    def test_no_valid_images_raises_value_error(self):
        with self.assertRaises(ValueError):
            Predictor(input_dir=str(self.invalid_files), output_dir=str(self.out_dir))