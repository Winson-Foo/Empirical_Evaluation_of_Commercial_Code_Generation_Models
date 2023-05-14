import os
import unittest

import numpy as np
import yaml
from tensorflow.keras.optimizers import Adam

from ISR.models.rrdn import RRDN
from ISR.models.rdn import RDN
from ISR.models.discriminator import Discriminator
from ISR.models.cut_vgg19 import Cut_VGG19


class ModelsClassTest(unittest.TestCase):
    def setUp(self):
        self.setup_config = yaml.safe_load(open(os.path.join('tests', 'data', 'config.yml'), 'r'))
        
        self.weights_path = {
            'generator': os.path.join(self.setup_config['weights_dir'], 'test_gen_weights.hdf5'),
            'discriminator': os.path.join(self.setup_config['weights_dir'], 'test_dis_weights.hdf5'),
        }
        
        self.hr_shape = (self.setup_config['patch_size'] * 2,) * 2 + (3,)
        
        self.rrdn_module = RrdnTestModule(self.setup_config, self.hr_shape)
        self.rdn_module = RdnTestModule(self.setup_config, self.hr_shape)
        self.f_ext_module = FeatureExtractorTestModule(self.setup_config)
        self.discr_module = DiscriminatorTestModule(self.setup_config, self.hr_shape)

    def test_RRDN_output_shapes(self):
        self.rrdn_module.assert_output_shapes()
        
    def test_RDN_output_shapes(self):
        self.rdn_module.assert_output_shapes()
    
    def test_trainable_layers_change(self):
        self.rrdn_module.assert_trainable_layers_change()
        self.rdn_module.assert_trainable_layers_change()
        self.discr_module.assert_trainable_layers_change()
    
    def test_feature_extractor_is_not_trainable(self):
        self.f_ext_module.assert_feature_extractor_not_trainable()
    
    
class RrdnTestModule:
    def __init__(self, setup_config, hr_shape):
        self.RRDN = RRDN(arch_params=setup_config['rrdn'], patch_size=setup_config['patch_size'])
        self.RRDN.model.compile(optimizer=Adam(), loss=['mse'])
        self.hr_shape = hr_shape
    
    def assert_output_shapes(self):
        self.assertTrue(self.RRDN.model.output_shape[1:4] == self.hr_shape)
    
    def assert_trainable_layers_change(self):
        x = np.random.random((1, self.setup_config['patch_size'], self.setup_config['patch_size'], 3))
        y = np.random.random((1, self.setup_config['patch_size'] * 2, self.setup_config['patch_size'] * 2, 3))
        self.assert_trainable_layers_change_on_batch(self.RRDN, x, y)

    
class RdnTestModule:
    def __init__(self, setup_config, hr_shape):
        self.RDN = RDN(arch_params=setup_config['rdn'], patch_size=setup_config['patch_size'])
        self.RDN.model.compile(optimizer=Adam(), loss=['mse'])
        self.hr_shape = hr_shape
        
    def assert_output_shapes(self):
        self.assertTrue(self.RDN.model.output_shape[1:4] == self.hr_shape)
    
    def assert_trainable_layers_change(self):
        x = np.random.random((1, self.setup_config['patch_size'], self.setup_config['patch_size'], 3))
        y = np.random.random((1, self.setup_config['patch_size'] * 2, self.setup_config['patch_size'] * 2, 3))
        self.assert_trainable_layers_change_on_batch(self.RDN, x, y)
    

class FeatureExtractorTestModule:
    VGG19_LAYERS_TO_EXTRACT = [1, 2]
    
    def __init__(self, setup_config):
        self.f_ext = Cut_VGG19(patch_size=setup_config['patch_size'] * 2, layers_to_extract=self.VGG19_LAYERS_TO_EXTRACT)
        self.f_ext.model.compile(optimizer=Adam(), loss=['mse', 'mse'])
    
    def assert_feature_extractor_not_trainable(self):
        y = np.random.random((1, self.setup_config['patch_size'] * 2, self.setup_config['patch_size'] * 2, 3))
        f_ext_out_shape = list(self.f_ext.model.outputs[0].shape[1:4])
        f_ext_out_shape1 = list(self.f_ext.model.outputs[1].shape[1:4])
        feats = [np.random.random([1] + f_ext_out_shape), np.random.random([1] + f_ext_out_shape1)]
        w_before = []
        for layer in self.f_ext.model.layers:
            if layer.trainable:
                w_before.append(layer.get_weights()[0])
        self.f_ext.model.train_on_batch(y, [*feats])
        for i, layer in enumerate(self.f_ext.model.layers):
            if layer.trainable:
                self.assertFalse(w_before[i] == layer.get_weights()[0])
    
    
class DiscriminatorTestModule:
    def __init__(self, setup_config, hr_shape):
        self.discr = Discriminator(patch_size=setup_config['patch_size'] * 2)
        self.discr.model.compile(optimizer=Adam(), loss=['mse'])
        self.hr_shape = hr_shape

    def assert_trainable_layers_change(self):
        x = np.random.random((1, self.setup_config['patch_size'] * 2, self.setup_config['patch_size'] * 2, 3))
        discr_out_shape = list(self.discr.model.outputs[0].shape)[1:4]
        valid = np.ones([1] + discr_out_shape)
        
        before_step = []
        for layer in self.discr.model.layers:
            if len(layer.trainable_weights) > 0:
                before_step.append(layer.get_weights()[0])
        
        self.discr.model.train_on_batch(x, valid)
        
        i = 0
        for layer in self.discr.model.layers:
            if len(layer.trainable_weights) > 0:
                self.assertFalse(np.all(before_step[i] == layer.get_weights()[0]))
                i += 1
    
    def assert_output_shapes(self):
        self.assertTrue(self.discr.model.output_shape[1:4] == self.hr_shape)
    
    def assert_trainable_layers_change_on_batch(self, model, x, y):
        
        before_step = []
        for layer in model.model.layers:
            if len(layer.trainable_weights) > 0:
                before_step.append(layer.get_weights()[0])
        
        model.model.train_on_batch(x, y)
        
        i = 0
        for layer in model.model.layers:
            if len(layer.trainable_weights) > 0:
                self.assertFalse(np.all(before_step[i] == layer.get_weights()[0]))
                i += 1)