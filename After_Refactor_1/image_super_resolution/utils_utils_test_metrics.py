import unittest

import numpy as np
import tensorflow.keras.backend as K

from ISR.utils.metrics import PSNR


class MetricsClassTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_psnr_sanity(self):
        # Arrange
        original_image = K.ones((10, 10, 3))
        distorted_image = K.zeros((10, 10, 3))
        
        # Act
        psnr_original = self._calculate_psnr(original_image, original_image)
        psnr_distorted = self._calculate_psnr(original_image, distorted_image)
        
        # Assert
        self.assertEqual(psnr_original, np.inf)
        self.assertEqual(psnr_distorted, 0)

    def _calculate_psnr(self, original_image, distorted_image):
        psnr = PSNR(original_image, distorted_image)
        try:
            return K.get_value(psnr)
        except:
            raise ValueError("Error calculating PSNR metric") 

if __name__ == '__main__':
    unittest.main()