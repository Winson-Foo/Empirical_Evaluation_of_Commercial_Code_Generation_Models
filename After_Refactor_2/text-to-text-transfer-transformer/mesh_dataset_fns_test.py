import unittest
from .mesh_dataset_fns import create_mesh_train_dataset, create_mesh_eval_dataset

class MeshDatasetFnsTest(unittest.TestCase):

    def test_create_mesh_train_dataset(self):
        mixture_name = "cached_mixture"
        sequence_length = {"inputs": 13, "targets": 13}
        ds = create_mesh_train_dataset(mixture_name, sequence_length)
        self.assertIsNotNone(ds)

    def test_create_mesh_eval_dataset(self):
        mixture_name = "uncached_mixture"
        sequence_length = {"inputs": 13, "targets": 13}

        # Test with cached dataset
        ds_cached = create_mesh_eval_dataset(mixture_name, sequence_length, use_cached=True)
        self.assertIsNotNone(ds_cached)

        # Test with uncached dataset
        ds_uncached = create_mesh_eval_dataset(mixture_name, sequence_length, use_cached=False)
        self.assertIsNotNone(ds_uncached)