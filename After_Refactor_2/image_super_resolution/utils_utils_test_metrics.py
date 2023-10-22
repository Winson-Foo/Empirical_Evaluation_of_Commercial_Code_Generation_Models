import unittest
import numpy as np
import tensorflow.keras.backend as K
from ISR.utils.metrics import PSNR

class MetricsClassTest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.image_shape = (10, 10, 3)
        cls.image_A = K.ones(shape=cls.image_shape)
        cls.image_B = K.zeros(shape=cls.image_shape)
    
    @classmethod
    def tearDownClass(cls):
        pass
    
    def test_PSNR_sanity(self):
        self.assertEqual(K.get_value(PSNR(self.image_A, self.image_A)), np.inf)
        self.assertEqual(K.get_value(PSNR(self.image_A, self.image_B)), 0)