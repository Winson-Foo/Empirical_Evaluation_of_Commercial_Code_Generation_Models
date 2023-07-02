import unittest
import torch
from nflows.transforms import permutations
from tests.transforms.transform_test import TransformTest

class PermutationTestCase(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10
        self.features = 100
        self.inputs = torch.randn(self.batch_size, self.features)
    
    def test_permutation(self, transform):
        transform = transform(self.features)
        outputs, logabsdet = transform(self.inputs)
        self.assertTensorIsGood(outputs, [self.batch_size, self.features])
        self.assertTensorIsGood(logabsdet, [self.batch_size])
        self.assertEqual(outputs, self.inputs[:, transform.permutation])
        self.assertEqual(logabsdet, torch.zeros([self.batch_size]))

    def assertTensorIsGood(self, tensor, expected_shape):
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, expected_shape)

    def assertForwardInverseAreConsistent(self, transform):
        temp, _ = transform(self.inputs)
        outputs, logabsdet = transform.inverse(temp)
        self.assertTensorIsGood(outputs, [self.batch_size, self.features])
        self.assertTensorIsGood(logabsdet, [self.batch_size])
        self.assertEqual(outputs, self.inputs)
        self.assertEqual(logabsdet, torch.zeros([self.batch_size]))

class PermutationTest(PermutationTestCase):
    def test_forward(self):
        self.test_permutation(permutations.Permutation)

    def test_inverse(self):
        self.test_permutation(permutations.Permutation)

    def test_forward_inverse_are_consistent(self):
        transforms = [
            permutations.Permutation,
            permutations.RandomPermutation,
            permutations.ReversePermutation,
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                self.assertForwardInverseAreConsistent(transform)

if __name__ == "__main__":
    unittest.main()