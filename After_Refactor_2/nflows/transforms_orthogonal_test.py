import unittest
import torch

from nflows.transforms import orthogonal
from nflows.utils import torchutils
from tests.transforms.transform_test import TransformTest


class HouseholderSequenceTest(TransformTest):
    def test_forward(self):
        """
        Test the forward method of the HouseholderSequence class.
        """
        features = 100
        batch_size = 50

        num_transforms_list = [1, 2, 11, 12]
        for num_transforms in num_transforms_list:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(features=features, num_transforms=num_transforms)
                matrix = transform.matrix()
                inputs = torch.randn(batch_size, features)
                outputs, logabsdet = transform(inputs)

                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.eps = 1e-5

                self.assertEqual(outputs, inputs @ matrix.t())
                self.assertEqual(logabsdet, torchutils.logabsdet(matrix) * torch.ones(batch_size))

    def test_inverse(self):
        """
        Test the inverse method of the HouseholderSequence class.
        """
        features = 100
        batch_size = 50

        num_transforms_list = [1, 2, 11, 12]
        for num_transforms in num_transforms_list:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(features=features, num_transforms=num_transforms)
                matrix = transform.matrix()
                inputs = torch.randn(batch_size, features)
                outputs, logabsdet = transform.inverse(inputs)

                self.assert_tensor_is_good(outputs, [batch_size, features])
                self.assert_tensor_is_good(logabsdet, [batch_size])
                self.eps = 1e-5

                self.assertEqual(outputs, inputs @ matrix)
                self.assertEqual(logabsdet, torchutils.logabsdet(matrix) * torch.ones(batch_size))

    def test_matrix(self):
        """
        Test the matrix method of the HouseholderSequence class.
        """
        features = 100

        num_transforms_list = [1, 2, 11, 12]
        for num_transforms in num_transforms_list:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(features=features, num_transforms=num_transforms)
                matrix = transform.matrix()

                self.assert_tensor_is_good(matrix, [features, features])
                self.eps = 1e-5

                self.assertEqual(matrix @ matrix.t(), torch.eye(features, features))
                self.assertEqual(matrix.t() @ matrix, torch.eye(features, features))
                self.assertEqual(matrix.t(), torch.inverse(matrix))
                det_ref = torch.tensor(1.0 if num_transforms % 2 == 0 else -1.0)
                self.assertEqual(matrix.det(), det_ref)

    def test_forward_inverse_are_consistent(self):
        """
        Test if the output of the forward and inverse methods are consistent for different transforms.
        """
        features = 100
        batch_size = 50
        inputs = torch.randn(batch_size, features)
        num_transforms_list = [1, 2, 11, 12]
        transforms = [orthogonal.HouseholderSequence(features=features, num_transforms=num_transforms)
                      for num_transforms in num_transforms_list]
        self.eps = 1e-5

        for transform in transforms:
            with self.subTest(transform=transform):
                self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()