import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import yaml

from ISR.models.rrdn import RRDN
from ISR.models.discriminator import Discriminator
from ISR.models.cut_vgg19 import Cut_VGG19
from ISR.utils.train_helper import TrainerHelper


CONFIG_PATH = './tests/data/config.yml'
WEIGHTS_DIR = 'tests/temporary_test_data/weights'
LOGS_DIR = 'tests/temporary_test_data/logs'
SESSION_ID = '0000'
DATANAME = 'TEST'


class UtilsClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONFIG_PATH) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        cls.rrdn = RRDN(arch_params=config['rrdn'], patch_size=config['patch_size'])
        cls.f_ext = Cut_VGG19(patch_size=config['patch_size'], layers_to_extract=[1, 2])
        cls.discr = Discriminator(patch_size=config['patch_size'])
        cls.weights_path = {
            'generator': Path(WEIGHTS_DIR) / 'test_gen_weights.hdf5',
            'discriminator': Path(WEIGHTS_DIR) / 'test_dis_weights.hdf5',
        }
        cls.th = TrainerHelper(
            generator=cls.rrdn,
            weights_dir=WEIGHTS_DIR,
            logs_dir=LOGS_DIR,
            lr_train_dir=config['lr_input'],
            feature_extractor=cls.f_ext,
            discriminator=cls.discr,
            dataname=DATANAME,
            weights_generator='',
            weights_discriminator='',
            fallback_save_every_n_epochs=2,
        )
        cls.th.session_id = SESSION_ID
        cls.th.logger.setLevel(50)

    def tearDown(self):
        if Path(WEIGHTS_DIR).exists():
            shutil.rmtree(WEIGHTS_DIR)
        if Path(LOGS_DIR).exists():
            shutil.rmtree(LOGS_DIR)

    def test_make_basename(self):
        expected_basename = 'rrdn-C2-D3-G20-G020-T2-x2'
        actual_basename = self.th._make_basename()
        self.assertEqual(actual_basename, expected_basename)

    def test_make_basename_with_pretrained_weights(self):
        expected_basename = 'rrdn-C2-D3-G20-G020-T2-x2'
        self.th.pretrained_weights_path = self.weights_path
        actual_basename = self.th._make_basename()
        self.th.pretrained_weights_path = {}
        self.assertEqual(actual_basename, expected_basename)

    def test_callback_paths_creation(self):
        expected_weights_path = Path(f"{WEIGHTS_DIR}/rrdn-C2-D3-G20-G020-T2-x2/{SESSION_ID}")
        expected_logs_path = Path(f"{LOGS_DIR}/rrdn-C2-D3-G20-G020-T2-x2/{SESSION_ID}")

        self.th.callback_paths = self.th._make_callback_paths()

        self.assertEqual(self.th.callback_paths['weights'], expected_weights_path)
        self.assertEqual(self.th.callback_paths['logs'], expected_logs_path)

    def test_weights_naming(self):
        expected_names = {
            'generator': Path(f"{WEIGHTS_DIR}/rrdn-C2-D3-G20-G020-T2-x2/{SESSION_ID}/rrdn-C2-D3-G20-G020-T2-x2{{metric}}_epoch{{epoch:03d}}.hdf5"),
            'discriminator': Path(f"{WEIGHTS_DIR}/rrdn-C2-D3-G20-G020-T2-x2/{SESSION_ID}/srgan-large{{metric}}_epoch{{epoch:03d}}.hdf5"),
        }

        self.th.callback_paths = self.th._make_callback_paths()
        actual_names = self.th._weights_name(self.th.callback_paths)

        self.assertEqual(actual_names['generator'], expected_names['generator'])
        self.assertEqual(actual_names['discriminator'], expected_names['discriminator'])

    def test_mock_training_setting_printer(self):
        with patch('ISR.utils.train_helper.TrainerHelper.print_training_setting', return_value=True):
            self.assertTrue(self.th.print_training_setting())

    def test_mock_epoch_end(self):
        with patch('ISR.utils.train_helper.TrainerHelper.on_epoch_end', return_value=True):
            self.assertTrue(self.th.on_epoch_end())

    def test_epoch_number_from_weights_names(self):
        weights_names = {
            'generator': 'test_gen_weights_TEST-vgg19-1-2-srgan-large-e003.hdf5',
            'discriminator': 'txxxxxxxxepoch003xxxxxhdf5',
            'discriminator2': 'test_discr_weights_TEST-vgg19-1-2-srgan-large-epoch03.hdf5',
        }
        self.assertEqual(self.th.epoch_n_from_weights_name(weights_names['generator']), 0)
        self.assertEqual(self.th.epoch_n_from_weights_name(weights_names['discriminator']), 3)
        self.assertEqual(self.th.epoch_n_from_weights_name(weights_names['discriminator2']), 0)

    def test_weights_saving(self):
        weights_dir = Path(f"{WEIGHTS_DIR}/rrdn-C2-D3-G20-G020-T2-x2/{SESSION_ID}")
        generator_weights_path = weights_dir / 'rrdn-C2-D3-G20-G020-T2-x2_epoch002.hdf5'
        discriminator_weights_path = weights_dir / 'srgan-large_epoch002.hdf5'

        self.th.callback_paths = self.th._make_callback_paths()
        self.th.weights_name = self.th._weights_name(self.th.callback_paths)

        weights_dir.mkdir(parents=True, exist_ok=True)

        self.th._save_weights(2, self.th.generator.model, self.th.discriminator, best=False)

        self.assertTrue(generator_weights_path.exists())
        self.assertTrue(discriminator_weights_path.exists())

    def test_training(self):
        with patch('ISR.utils.train_helper.TrainerHelper.initialize_training', return_value=True):
            with patch('ISR.utils.train_helper.TrainerHelper.on_epoch_end', return_value=True):
                with patch('ISR.utils.train_helper.TrainerHelper.print_training_setting', return_value=True):
                    weights_dir = Path(f"{WEIGHTS_DIR}/rrdn-C2-D3-G20-G020-T2-x2/{SESSION_ID}")
                    weights_dir.mkdir(parents=True)
                    self.th.train(1)
                    self.assertTrue(Path(f"{LOGS_DIR}/train_log_TEST.csv").exists())
                    self.assertTrue(weights_dir.exists())