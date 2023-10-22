import logging
import unittest
import shutil
import yaml
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch, Mock

from ISR.models.rdn import RDN
from ISR.predict.predictor import Predictor


class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)
        cls.config = yaml.safe_load((Path('tests/data/config.yml').read_text()))
        cls.model = RDN(arch_params=cls.config['rdn'], patch_size=cls.config['patch_size'])
        cls.input_dir = Path('tests/temporary_test_data/valid_files')
        cls.output_dir = Path('tests/temporary_test_data/out_dir/valid_files')
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        cls.file_paths = [cls.input_dir / fn for fn in ['data2.gif', 'data1.png', 'data0.jpeg']]
        for file_path in cls.file_paths:
            file_path.touch()
        
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(Path('tests/temporary_test_data'))
    
    def setUp(self):
        self.predictor = Predictor(input_dir=str(self.input_dir), output_dir=str(self.output_dir))
        self.predictor.logger = Mock(return_value=True)
    
    def tearDown(self):
        pass
    
    def test_load_weights_with_no_weights(self):
        self.predictor.weights_path = None
        with self.assertRaises(ValueError):
            self.predictor._load_weights()
    
    def test_load_weights_with_valid_weights(self):
        def raise_path(path):
            raise ValueError(path)
        
        self.predictor.model = self.model
        self.predictor.model.model.load_weights = Mock(side_effect=raise_path)
        self.predictor.weights_path = 'a/path'
        with self.assertRaises(ValueError) as cm:
            self.predictor._load_weights()
        self.assertEqual(str(cm.exception), 'a/path')
    
    def test_make_basename(self):
        self.predictor.model = self.model
        made_name = self.predictor._make_basename()
        self.assertEqual(made_name, 'rdn-C3-D10-G64-G064-x2')
    
    def test_forward_pass_pixel_range_and_type(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr
        
        self.predictor.model = self.model
        self.predictor.model.model.predict = Mock(side_effect=valid_sr_output)
        with patch('imageio.imread', return_value=np.random.random((10, 10, 3))):
            sr = self.predictor._forward_pass(str(self.file_paths[0]))
        self.assertIsInstance(sr, np.ndarray)
        self.assertEqual(sr.dtype, np.uint8)
        self.assertTrue(np.all(sr >= 0.0))
        self.assertTrue(np.all(sr <= 255.0))
        self.assertTrue(np.any(sr > 1.0))
        self.assertEqual(sr.shape, (20, 20, 3))
    
    def test_forward_pass_4_channels(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr
        
        self.predictor.model = self.model
        self.predictor.model.model.predict = Mock(side_effect=valid_sr_output)
        with patch('imageio.imread', return_value=np.random.random((10, 10, 4))):
            sr = self.predictor._forward_pass(str(self.file_paths[0]))
        self.assertIsNone(sr)
    
    def test_forward_pass_1_channel(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr
        
        self.predictor.model = self.model
        self.predictor.model.model.predict = Mock(side_effect=valid_sr_output)
        with patch('imageio.imread', return_value=np.random.random((10, 10, 1))):
            sr = self.predictor._forward_pass(str(self.file_paths[0]))
        self.assertIsNone(sr)
    
    def test_get_predictions(self):
        self.predictor._load_weights = Mock(return_value={})
        self.predictor._forward_pass = Mock(return_value=True)
        with patch('imageio.imwrite', return_value=True):
            self.predictor.get_predictions(self.model, 'a/path/arch-weights_session1_session2.hdf5')
    
    def test_output_folder_and_dataname(self):
        self.assertEqual(self.predictor.data_name, 'valid_files')
        self.assertEqual(self.predictor.output_dir, Path('tests/temporary_test_data/out_dir/valid_files'))
    
    def test_valid_extensions(self):
        valid_files = sorted([self.input_dir / 'data0.jpeg', self.input_dir / 'data1.png'])
        self.assertEqual(sorted(self.predictor.img_ls), valid_files)
    
    def test_no_valid_images(self):
        with self.assertRaisesRegex(ValueError, 'No valid images in input directory'):
            Predictor(input_dir=str(Path('tests/temporary_test_data/invalid_files')), output_dir='out_dir')