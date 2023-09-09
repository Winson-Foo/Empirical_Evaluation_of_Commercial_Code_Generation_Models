import os
import unittest
import yaml
import numpy as np
from tensorflow.keras.optimizers import Adam
from ISR.models.rrdn import RRDN
from ISR.models.rdn import RDN
from ISR.models.discriminator import Discriminator
from ISR.models.cut_vgg19 import Cut_VGG19

CONFIG_FILE = os.path.join('tests', 'data', 'config.yml')

class ModelsClassTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = yaml.load(open(CONFIG_FILE, 'r'))
        cls.weights_path = {
            'generator': os.path.join(cls.setup['weights_dir'], 'test_gen_weights.hdf5'),
            'discriminator': os.path.join(cls.setup['weights_dir'], 'test_dis_weights.hdf5'),
        }
        cls.hr_shape = (cls.setup['patch_size'] * 2,) * 2 + (3,)
        cls.RRDN = RRDN(arch_params=cls.setup['rrdn'], patch_size=cls.setup['patch_size'])
        cls.RDN = RDN(arch_params=cls.setup['rdn'], patch_size=cls.setup['patch_size'])
        cls.f_ext = Cut_VGG19(patch_size=cls.setup['patch_size'] * 2, layers_to_extract=[1, 2])
        cls.discr = Discriminator(patch_size=cls.setup['patch_size'] * 2)
        cls._compile_models()
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_SR_output_shapes(self):
        self.assertTrue(self.RRDN.model.output_shape[1:4] == self.hr_shape)
        self.assertTrue(self.RDN.model.output_shape[1:4] == self.hr_shape)
    
    def test_that_the_trainable_layers_change(self):
        x = np.random.random((1, self.setup['patch_size'], self.setup['patch_size'], 3))
        y = np.random.random((1, self.setup['patch_size'] * 2, self.setup['patch_size'] * 2, 3))
        before_step = []
        self._get_trainable_layers(self.RRDN, before_step)
        self.RRDN.model.train_on_batch(x, y)
        i = 0
        for layer in self.RRDN.model.layers:
            if len(layer.trainable_weights) > 0:
                self.assertFalse(np.all(before_step[i] == layer.get_weights()[0]))
                i += 1
        
        before_step = []
        self._get_trainable_layers(self.RDN, before_step)
        self.RDN.model.train_on_batch(x, y)
        i = 0
        for layer in self.RDN.model.layers:
            if len(layer.trainable_weights) > 0:
                self.assertFalse(np.all(before_step[i] == layer.get_weights()[0]))
                i += 1
        
        discr_out_shape = list(self.discr.model.outputs[0].shape)[1:4]
        valid = np.ones([1] + discr_out_shape)
        
        before_step = []
        self._get_trainable_layers(self.discr, before_step)
        self.discr.model.train_on_batch(y, valid)
        
        i = 0
        for layer in self.discr.model.layers:
            if len(layer.trainable_weights) > 0:
                self.assertFalse(np.all(before_step[i] == layer.get_weights()[0]))
                i += 1
    
    def test_that_feature_extractor_is_not_trainable(self):
        y = np.random.random((1, self.setup['patch_size'] * 2, self.setup['patch_size'] * 2, 3))
        f_ext_out_shape = list(self.f_ext.model.outputs[0].shape[1:4])
        f_ext_out_shape1 = list(self.f_ext.model.outputs[1].shape[1:4])
        feats = [np.random.random([1] + f_ext_out_shape), np.random.random([1] + f_ext_out_shape1)]
        w_before = []
        self._get_trainable_layers(self.f_ext, w_before)
        self.f_ext.model.train_on_batch(y, [*feats])
        for i, layer in enumerate(self.f_ext.model.layers):
            if layer.trainable:
                self.assertFalse(w_before[i] == layer.get_weights()[0])
    
    @classmethod
    def _get_trainable_layers(cls, model, layers):
        for layer in model.model.layers:
            if layer.trainable:
                weights = layer.get_weights()[0]
                layers.append(weights)
    
    @classmethod
    def _compile_models(cls):
        cls.RRDN.model.compile(optimizer=Adam(), loss=['mse'])
        cls.RDN.model.compile(optimizer=Adam(), loss=['mse'])
        cls.f_ext.model.compile(optimizer=Adam(), loss=['mse', 'mse'])
        cls.discr.model.compile(optimizer=Adam(), loss=['mse'])