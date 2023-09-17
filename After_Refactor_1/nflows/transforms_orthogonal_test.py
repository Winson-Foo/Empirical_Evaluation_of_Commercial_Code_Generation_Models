import unittest
import torch
from nflows.transforms import orthogonal
from nflows.utils import torchutils
from tests.transforms.transform_test import TransformTest

class HouseholderSequenceTest(TransformTest):
    def setUp(self):
        self.features = 100
        self.batch_size = 50
        self.num_transforms = [1, 2, 11, 12]
        self.inputs = torch.randn(self.batch_size, self.features)

    def test_forward(self):
        for num_transforms in self.num_transforms:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=self.features, num_transforms=num_transforms
                )
                matrix = transform.matrix()
                outputs, logabsdet = transform(self.inputs)
                self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
                self.assert_tensor_is_good(logabsdet, [self.batch_size])
                self.eps = 1e-5
                self.assertEqual(outputs, self.inputs @ matrix.t())
                self.assertEqual(
                    logabsdet, torchutils.logabsdet(matrix) * torch.ones(self.batch_size)
                )

    def test_inverse(self):
        for num_transforms in self.num_transforms:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=self.features, num_transforms=num_transforms
                )
                matrix = transform.matrix()
                outputs, logabsdet = transform.inverse(self.inputs)
                self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
                self.assert_tensor_is_good(logabsdet, [self.batch_size])
                self.eps = 1e-5
                self.assertEqual(outputs, self.inputs @ matrix)
                self.assertEqual(
                    logabsdet, torchutils.logabsdet(matrix) * torch.ones(self.batch_size)
                )

    def test_matrix(self):
        for num_transforms in self.num_transforms:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=self.features, num_transforms=num_transforms
                )
                matrix = transform.matrix()
                self.assert_tensor_is_good(matrix, [self.features, self.features])
                self.eps = 1e-5
                self.assertEqual(matrix @ matrix.t(), torch.eye(self.features, self.features))
                self.assertEqual(matrix.t() @ matrix, torch.eye(self.features, self.features))
                self.assertEqual(matrix.t(), torch.inverse(matrix))
                det_ref = torch.tensor(1.0 if num_transforms % 2 == 0 else -1.0)
                self.assertEqual(matrix.det(), det_ref)

    def test_forward_inverse_are_consistent(self):
        self.eps = 1e-5
        for num_transforms in self.num_transforms:
            with self.subTest(num_transforms=num_transforms):
                transform = orthogonal.HouseholderSequence(
                    features=self.features, num_transforms=num_transforms
                )
                self.assert_forward_inverse_are_consistent(transform, self.inputs)


if __name__ == "__main__":
    unittest.main()