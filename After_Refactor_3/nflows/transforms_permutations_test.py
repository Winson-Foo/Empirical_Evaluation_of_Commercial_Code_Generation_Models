import unittest

import torch

from nflows.transforms import permutations
from tests.transforms.transform_test import TransformTest


class PermutationTest(TransformTest):
    def setUp(self):
        self.batch_size = 10
        self.features = 100

    def create_random_inputs(self):
        return torch.randn(self.batch_size, self.features)

    def create_random_permutation(self):
        return torch.randperm(self.features)

    def assert_outputs_consistent(self, transform, inputs, outputs, logabsdet):
        self.assert_tensor_is_good(outputs, [self.batch_size, self.features])
        self.assert_tensor_is_good(logabsdet, [self.batch_size])
        self.assertEqual(outputs, inputs[:, transform.permutation])
        self.assertEqual(logabsdet, torch.zeros([self.batch_size]))

    def test_forward(self):
        inputs = self.create_random_inputs()
        permutation = self.create_random_permutation()
        transform = permutations.Permutation(permutation)
        outputs, logabsdet = transform(inputs)
        self.assert_outputs_consistent(transform, inputs, outputs, logabsdet)

    def test_inverse(self):
        inputs = self.create_random_inputs()
        permutation = self.create_random_permutation()
        transform = permutations.Permutation(permutation)
        temp, _ = transform(inputs)
        outputs, logabsdet = transform.inverse(temp)
        self.assert_outputs_consistent(transform, inputs, outputs, logabsdet)

    def test_forward_inverse_are_consistent(self):
        inputs = self.create_random_inputs()
        transforms = [
            permutations.Permutation(self.create_random_permutation()),
            permutations.RandomPermutation(self.features),
            permutations.ReversePermutation(self.features),
        ]
        for transform in transforms:
            with self.subTest(transform=transform):
                self.assert_forward_inverse_are_consistent(transform, inputs)


if __name__ == "__main__":
    unittest.main()