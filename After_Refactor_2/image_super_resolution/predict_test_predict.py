import logging
import shutil
from copy import copy
from pathlib import Path
from unittest.mock import patch, Mock
import yaml
import numpy as np
import unittest
from ISR.models.rdn import RDN
from ISR.predict.predictor import Predictor


class PredictorClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.disable(logging.CRITICAL)

        # Load configuration setup from YAML file
        with open('tests/data/config.yml') as f:
            cls.setup = yaml.safe_load(f)

        # Initialize RDN model
        cls.RDN = RDN(arch_params=cls.setup['rdn'], patch_size=cls.setup['patch_size'])

        # Create temporary directories for testing
        cls.temp_data = Path('tests/temporary_test_data')
        cls.temp_data.mkdir(parents=True, exist_ok=True)

        cls.valid_files = cls.temp_data / 'valid_files'
        cls.valid_files.mkdir(parents=True, exist_ok=True)
        for item in ['data2.gif', 'data1.png', 'data0.jpeg']:
            (cls.valid_files / item).touch()

        cls.invalid_files = cls.temp_data / 'invalid_files'
        cls.invalid_files.mkdir(parents=True, exist_ok=True)
        for item in ['data2.gif', 'data.data', 'data02']:
            (cls.invalid_files / item).touch()

        # Set up mock logger and output directory
        cls.out_dir = cls.temp_data / 'out_dir'
        cls.predictor = Predictor(input_dir=str(cls.valid_files), output_dir=str(cls.out_dir))
        cls.predictor.logger = Mock(return_value=True)

    @classmethod
    def tearDownClass(cls):
        # Delete temporary directories after testing is complete
        shutil.rmtree(cls.temp_data)

    def setUp(self):
        # Create copy of predictor for testing
        self.pred = copy(self.predictor)

    def tearDown(self):
        pass

    # Test cases
    def test_load_weights_with_no_weights(self):
        self.pred.weights_path = None
        with self.assertRaises(ValueError):
            self.pred._load_weights()

    def test_load_weights_with_valid_weights(self):
        def raise_path(path):
            raise ValueError(path)

        self.pred.model = self.RDN
        self.pred.model.model.load_weights = Mock(side_effect=raise_path)
        self.pred.weights_path = 'a/path'
        with self.assertRaises(ValueError) as e:
            self.pred._load_weights()
            self.assertEqual(str(e), 'a/path')

    def test_make_basename(self):
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

        # Assert that super-resolution output is valid
        self.assertEqual(type(sr[0, 0, 0]), np.uint8)
        self.assertTrue(np.all(sr >= 0.0))
        self.assertTrue(np.all(sr <= 255.0))
        self.assertTrue(np.any(sr > 1.0))
        self.assertEqual(sr.shape, (20, 20, 3))

    def test_forward_pass_4_channels(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr

        self.pred.model = self.RDN
        self.pred.model.model.predict = Mock(side_effect=valid_sr_output)
        with patch('imageio.imread', return_value=np.random.random((10, 10, 4))):
            sr = self.pred._forward_pass('file_path')

        # Assert that super-resolution output is None for 4-channel input
        self.assertIsNone(sr)

    def test_forward_pass_1_channel(self):
        def valid_sr_output(*args):
            sr = np.random.random((1, 20, 20, 3))
            sr[0, 0, 0, 0] = 0.5
            return sr

        self.pred.model = self.RDN
        self.pred.model.model.predict = Mock(side_effect=valid_sr_output)
        with patch('imageio.imread', return_value=np.random.random((10, 10, 1))):
            sr = self.pred._forward_pass('file_path')

        # Assert that super-resolution output is None for 1-channel input
        self.assertIsNone(sr)

    def test_get_predictions(self):
        self.pred._load_weights = Mock(return_value={})
        self.pred._forward_pass = Mock(return_value=True)
        with patch('imageio.imwrite', return_value=True):
            self.pred.get_predictions(self.RDN, 'a/path/arch-weights_session1_session2.hdf5')

    def test_output_folder_and_dataname(self):
        self.assertEqual(self.pred.data_name, 'valid_files')
        self.assertEqual(self.pred.output_dir, Path(f'tests/temporary_test_data/out_dir/{self.pred.data_name}'))

    def test_valid_extensions(self):
        img_ls = np.sort(self.pred.img_ls)
        expected_output = np.sort([self.valid_files / 'data0.jpeg', self.valid_files / 'data1.png'])
        self.assertTrue(np.array_equal(img_ls, expected_output))

    def test_no_valid_images(self):
        with self.assertRaises(ValueError) as e:
            Predictor(input_dir=str(self.invalid_files), output_dir=str(self.out_dir))
            self.assertTrue('image' in str(e))