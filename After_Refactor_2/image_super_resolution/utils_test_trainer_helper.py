import unittest
import shutil
from pathlib import Path
from unittest.mock import patch

import yaml

from ISR.models.cut_vgg19 import Cut_VGG19
from ISR.models.discriminator import Discriminator
from ISR.models.rrdn import RRDN
from ISR.utils.train_helper import TrainerHelper


class UtilsClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the config file
        config_file = Path('./tests/data/config.yml')
        cls.setup = yaml.load(config_file.read_text())

        # Initialize the required objects
        cls.rrdn = RRDN(arch_params=cls.setup['rrdn'], patch_size=cls.setup['patch_size'])
        cls.cutvgg = Cut_VGG19(patch_size=cls.setup['patch_size'], layers_to_extract=[1, 2])
        cls.discriminator = Discriminator(patch_size=cls.setup['patch_size'])
        cls.weights_path = {
            'generator': Path(cls.setup['weights_dir']) / 'test_gen_weights.hdf5',
            'discriminator': Path(cls.setup['weights_dir']) / 'test_dis_weights.hdf5',
        }
        cls.trainer_helper = TrainerHelper(
            generator=cls.rrdn,
            weights_dir=cls.setup['weights_dir'],
            logs_dir=cls.setup['log_dir'],
            lr_train_dir=cls.setup['lr_input'],
            feature_extractor=cls.cutvgg,
            discriminator=cls.discriminator,
            dataname='TEST',
            weights_generator='',
            weights_discriminator='',
            fallback_save_every_n_epochs=2,
        )
        cls.trainer_helper.session_id = '0000'
        cls.trainer_helper.logger.setLevel(50)

    def tearDown(self):
        # Remove temporary files and directories created during testing
        if Path('./tests/temporary_test_data').exists():
            shutil.rmtree('./tests/temporary_test_data')
        if Path('./log_file').exists():
            Path('./log_file').unlink()

    def test__make_basename(self):
        # Test the _make_basename method of TrainerHelper
        generator_name = self.trainer_helper.generator.name + '-C2-D3-G20-G020-T2-x2'
        generated_name = self.trainer_helper._make_basename()
        self.assertEqual(generated_name, generator_name)

    def test_basename_without_pretrained_weights(self):
        # Test the _make_basename method of TrainerHelper when no pretrained weights are given
        basename = 'rrdn-C2-D3-G20-G020-T2-x2'
        self.trainer_helper.pretrained_weights_path = {}
        made_basename = self.trainer_helper._make_basename()
        self.assertEqual(made_basename, basename)

    def test_basename_with_pretrained_weights(self):
        # Test the _make_basename method of TrainerHelper when pretrained weights are given
        basename = 'rrdn-C2-D3-G20-G020-T2-x2'
        self.trainer_helper.pretrained_weights_path = self.weights_path
        made_basename = self.trainer_helper._make_basename()
        self.trainer_helper.pretrained_weights_path = {}
        self.assertEqual(made_basename, basename)

    def test_callback_paths_creation(self):
        # Test the _make_callback_paths method of TrainerHelper
        self.trainer_helper.callback_paths = self.trainer_helper._make_callback_paths()
        self.assertEqual(
            self.trainer_helper.callback_paths['weights'],
            Path('tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000'),
        )
        self.assertEqual(
            self.trainer_helper.callback_paths['logs'],
            Path('tests/temporary_test_data/logs/rrdn-C2-D3-G20-G020-T2-x2/0000'),
        )

    def test_weights_naming(self):
        # Test the _weights_name method of TrainerHelper
        w_names = {
            'generator': Path(
                'tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/rrdn-C2-D3-G20-G020-T2-x2{metric}_epoch{epoch:03d}.hdf5'
            ),
            'discriminator': Path(
                'tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/srgan-large{metric}_epoch{epoch:03d}.hdf5'
            ),
        }
        cb_paths = self.trainer_helper._make_callback_paths()
        generated_names = self.trainer_helper._weights_name(cb_paths)
        self.assertEqual(generated_names['generator'], w_names['generator'])
        self.assertEqual(generated_names['discriminator'], w_names['discriminator'])

    def test_weights_saving(self):
        # Test the _save_weights method of TrainerHelper
        self.trainer_helper.callback_paths = self.trainer_helper._make_callback_paths()
        self.trainer_helper.weights_name = self.trainer_helper._weights_name(self.trainer_helper.callback_paths)
        Path('tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/').mkdir(parents=True)
        self.trainer_helper._save_weights(1, self.trainer_helper.generator.model, self.trainer_helper.discriminator, best=False)

        self.assertTrue(
            Path('./tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/rrdn-C2-D3-G20-G020-T2-x2_epoch002.hdf5').exists()
        )
        self.assertTrue(
            Path('./tests/temporary_test_data/weights/rrdn-C2-D3-G20-G020-T2-x2/0000/srgan-large_epoch002.hdf5').exists()
        )

    def test_epoch_number_from_weights_names(self):
        # Test the epoch_n_from_weights_name method of TrainerHelper
        w_names = {
            'generator': 'test_gen_weights_TEST-vgg19-1-2-srgan-large-e003.hdf5',
            'discriminator': 'txxxxxxxxepoch003xxxxxhdf5',
            'discriminator2': 'test_discr_weights_TEST-vgg19-1-2-srgan-large-epoch03.hdf5',
        }
        e_n = self.trainer_helper.epoch_n_from_weights_name(w_names['generator'])
        self.assertEqual(e_n, 0)
        e_n = self.trainer_helper.epoch_n_from_weights_name(w_names['discriminator'])
        self.assertEqual(e_n, 3)
        e_n = self.trainer_helper.epoch_n_from_weights_name(w_names['discriminator2'])
        self.assertEqual(e_n, 0)


class MockTrainingSettingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_helper = TrainerHelper()

    def test_mock_training_setting_printer(self):
        # Test the print_training_setting method of TrainerHelper using a mock function
        with patch('ISR.utils.train_helper.TrainerHelper.print_training_setting', return_value=True):
            self.assertTrue(self.trainer_helper.print_training_setting())


class MockEpochEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_helper = TrainerHelper()

    def test_mock_epoch_end(self):
        # Test the on_epoch_end method of TrainerHelper using a mock function
        with patch('ISR.utils.train_helper.TrainerHelper.on_epoch_end', return_value=True):
            self.assertTrue(self.trainer_helper.on_epoch_end())


class MockTrainingInitializationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer_helper = TrainerHelper()

    def test_mock_initalize_training(self):
        # Test the initialize_training method of TrainerHelper using a mock function
        with patch('ISR.utils.train_helper.TrainerHelper.initialize_training', return_value=True):
            self.assertTrue(self.trainer_helper.initialize_training())