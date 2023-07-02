import unittest
import torch
from nflows.transforms import permutations
from tests.transforms.transform_test import TransformTest

class PermutationTest(TransformTest):
    def setUp(self):
        self.batch_size = 10
        self.features = 100
        self.inputs = torch.randn(self.batch_size, self.features)
        self.permutation = torch.randperm(self.features)

    def test_permutation_forward(self):
        transform = permutations.Permutation(self.permutation)
        outputs, logabsdet = transform(self.inputs)
        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])
        self.assertEqual(outputs, self.inputs[:, self.permutation])
        self.assertEqual(logabsdet, torch.zeros([self.batch_size]))

    def test_permutation_inverse(self):
        transform = permutations.Permutation(self.permutation)
        temp, _ = transform(self.inputs)
        outputs, logabsdet = transform.inverse(temp)
        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])
        self.assertEqual(outputs, self.inputs)
        self.assertEqual(logabsdet, torch.zeros([self.batch_size]))

    def test_forward_inverse_are_consistent(self):
        transforms = [
            permutations.Permutation(torch.randperm(self.features)),
            permutations.RandomPermutation(self.features),
            permutations.ReversePermutation(self.features),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                self.assert_forward_inverse_are_consistent(transform, self.inputs)

if __name__ == "__main__":
    unittest.main()