import os
import shutil
import unittest
from unittest.mock import patch

import yaml
from ISR.models.cut_vgg19 import Cut_VGG19
from ISR.models.discriminator import Discriminator
from ISR.models.rrdn import RRDN
from ISR.utils.train_helper import TrainerHelper


class UtilsClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open('./tests/data/config.yml', 'r') as f:
            cls.setup = yaml.load(f, Loader=yaml.FullLoader)
        cls.RRDN = RRDN(arch_params=cls.setup['rrdn'], patch_size=cls.setup['patch_size'])
        cls.f_ext = Cut_VGG19(patch_size=cls.setup['patch_size'], layers_to_extract=[1, 2])
        cls.discr = Discriminator(patch_size=cls.setup['patch_size'])
        cls.weights_dir = 'tests/temporary_test_data/weights'
        cls.logs_dir = 'tests/temporary_test_data/logs'
        cls.weights_path = {
            'generator': os.path.join(cls.weights_dir, 'test_gen_weights.hdf5'),
            'discriminator': os.path.join(cls.weights_dir, 'test_dis_weights.hdf5'),
        }
        cls.TH = TrainerHelper(
            generator=cls.RRDN,
            weights_dir=cls.weights_dir,
            logs_dir=cls.logs_dir,
            lr_train_dir=cls.setup['lr_input'],
            feature_extractor=cls.f_ext,
            discriminator=cls.discr,
            dataname='TEST',
            weights_generator='',
            weights_discriminator='',
            fallback_save_every_n_epochs=2,
        )
        cls.TH.session_id = '0000'
        cls.TH.logger.setLevel(50)

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        if os.path.exists('tests/temporary_test_data'):
            shutil.rmtree('tests/temporary_test_data')
        if os.path.exists('./log_file'):
            os.remove('./log_file')

    def test__make_basename(self):
        expected_name = '{}-C2-D3-G20-G020-T2-x2'.format(self.TH.generator.name)
        generated_name = self.TH._make_basename()
        self.assertEqual(generated_name, expected_name, 'Generated name: {}, expected: {}'.format(generated_name, expected_name))

    def test_basename_with_pretrained_weights(self):
        basename = 'rrdn-C2-D3-G20-G020-T2-x2'
        self.TH.pretrained_weights_path = self.weights_path
        made_basename = self.TH._make_basename()
        self.TH.pretrained_weights_path = {}
        self.assertEqual(basename, made_basename, 'Generated name: {}, expected: {}'.format(made_basename, basename))

    def test_basename_without_pretrained_weights(self):
        basename = 'rrdn-C2-D3-G20-G020-T2-x2'
        made_basename = self.TH._make_basename()
        self.assertEqual(basename, made_basename, 'Generated name: {}, expected: {}'.format(made_basename, basename))

    def test_callback_paths_creation(self):
        # Reset session_id
        self.TH.session_id = None
        self.TH.callback_paths = self.TH._make_callback_paths()
        expected_weights_path = os.path.join(self.weights_dir, 'rrdn-C2-D3-G20-G020-T2-x2', '0000')
        expected_logs_path = os.path.join(self.logs_dir, 'rrdn-C2-D3-G20-G020-T2-x2', '0000')
        self.assertEqual(self.TH.callback_paths['weights'], expected_weights_path,
                         'Generated path: {}, expected: {}'.format(self.TH.callback_paths['weights'], expected_weights_path)
                         )
        self.assertEqual(self.TH.callback_paths['logs'], expected_logs_path,
                         'Generated path: {}, expected: {}'.format(self.TH.callback_paths['logs'], expected_logs_path)
                         )

    def test_epoch_number_from_weights_names(self):
        expected_epoch_number = 3
        w_name = 'test_gen_weights_TEST-vgg19-1-2-srgan-large-e003.hdf5'
        epoch_number = self.TH.epoch_n_from_weights_name(w_name)
        self.assertEqual(epoch_number, expected_epoch_number,
                         'Expected: {}, got: {}'.format(expected_epoch_number, epoch_number))

    def test_mock_epoch_end(self):
        with patch('ISR.utils.train_helper.TrainerHelper.on_epoch_end', return_value=True):
            self.assertTrue(self.TH.on_epoch_end())

    def test_mock_initalize_training(self):
        with patch('ISR.utils.train_helper.TrainerHelper.initialize_training', return_value=True):
            self.assertTrue(self.TH.initialize_training())

    def test_mock_training_setting_printer(self):
        with patch('ISR.utils.train_helper.TrainerHelper.print_training_setting', return_value=True):
            self.assertTrue(self.TH.print_training_setting())

    def test_weights_naming(self):
        expected_generator_w_name = os.path.join(self.weights_dir, 'rrdn-C2-D3-G20-G020-T2-x2', '0000', 'rrdn-C2-D3-G20-G020-T2-x2{metric}_epoch{epoch:03d}.hdf5')
        expected_discriminator_w_name = os.path.join(self.weights_dir, 'rrdn-C2-D3-G20-G020-T2-x2', '0000', 'srgan-large{metric}_epoch{epoch:03d}.hdf5')
        self.TH.callback_paths = self.TH._make_callback_paths()
        generated_names = self.TH._weights_name(self.TH.callback_paths)
        self.assertEqual(generated_names['generator'], expected_generator_w_name,
                         'Generated name: {}, expected: {}'.format(generated_names['generator'], expected_generator_w_name))
        self.assertEqual(generated_names['discriminator'], expected_discriminator_w_name,
                         'Generated name: {}, expected: {}'.format(generated_names['discriminator'], expected_discriminator_w_name))

    def test_weights_saving(self):
        epoch_number = 2
        self.TH.callback_paths = {'weights': os.path.join(self.weights_dir, 'rrdn-C2-D3-G20-G020-T2-x2', '0000')}
        expected_generator_w_path = os.path.join(self.weights_dir, 'rrdn-C2-D3-G20-G020-T2-x2', '0000', 'rrdn-C2-D3-G20-G020-T2-x2_epoch002.hdf5')
        expected_discriminator_w_path = os.path.join(self.weights_dir, 'rrdn-C2-D3-G20-G020-T2-x2', '0000', 'srgan-large_epoch002.hdf5')

        if not os.path.exists(self.TH.callback_paths['weights']):
            os.makedirs(self.TH.callback_paths['weights'])

        self.TH._save_weights(epoch_number, self.TH.generator.model, self.TH.discriminator, best=False)

        self.assertTrue(os.path.exists(expected_generator_w_path),
                        'Expected file to exist: {}'.format(expected_generator_w_path))
        self.assertTrue(os.path.exists(expected_discriminator_w_path),
                        'Expected file to exist: {}'.format(expected_discriminator_w_path))