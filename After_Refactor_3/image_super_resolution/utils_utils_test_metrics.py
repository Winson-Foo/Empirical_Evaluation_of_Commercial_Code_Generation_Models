import unittest
import numpy as np
import tensorflow.keras.backend as K
from ISR.utils.metrics import PSNR


class MetricsClassTest(unittest.TestCase):
    """
    Tests for the ISR metrics module.
    """

    def test_PSNR_with_identical_images(self):
        """
        Test that PSNR is infinite when comparing identical images.
        """
        image = K.ones((10, 10, 3))
        self.assertAlmostEqual(
            K.get_value(PSNR(image, image)),
            np.inf,
            delta=1e-6,
            msg="PSNR was not infinite when comparing identical images."
        )

    def test_PSNR_with_different_images(self):
        """
        Test that PSNR is 0 when comparing two completely different images.
        """
        image1 = K.ones((10, 10, 3))
        image2 = K.zeros((10, 10, 3))
        self.assertAlmostEqual(
            K.get_value(PSNR(image1, image2)),
            0,
            delta=1e-6,
            msg="PSNR was not 0 when comparing completely different images."
        )

    def test_PSNR_with_noise(self):
        """
        Test that PSNR decreases with increasing noise.
        """
        image1 = K.ones((10, 10, 3))
        image2 = image1 + K.random_normal((10, 10, 3), mean=0.0, stddev=0.5)
        image3 = image1 + K.random_normal((10, 10, 3), mean=0.0, stddev=1.0)
        psnr1 = K.get_value(PSNR(image1, image2))
        psnr2 = K.get_value(PSNR(image1, image3))
        self.assertLess(
            psnr2,
            psnr1,
            msg="PSNR did not decrease with increasing noise."
        )

    def test_PSNR_with_scaling(self):
        """
        Test that PSNR is scale-invariant.
        """
        image1 = K.ones((10, 10, 3))
        image2 = image1 * 0.5
        self.assertAlmostEqual(
            K.get_value(PSNR(image1, image2)),
            K.get_value(PSNR(image2, image1)),
            delta=1e-6,
            msg="PSNR was not scale-invariant."
        )