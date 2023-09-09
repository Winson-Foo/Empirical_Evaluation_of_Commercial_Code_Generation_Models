import os
import unittest
import urllib

from shorttext.metrics.wasserstein import word_mover_distance
from shorttext.utils import load_word2vec_model


WORD_EMBEDDING_MODEL_FILENAME = "test_w2v_model.bin"
WORD_EMBEDDING_MODEL_URL = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"


class TestWMD(unittest.TestCase):
    def setUp(self):
        self.download_word_embedding_model()
        self.w2v_model = load_word2vec_model(WORD_EMBEDDING_MODEL_FILENAME, binary=True)

    def tearDown(self):
        self.remove_word_embedding_model()

    def download_word_embedding_model(self):
        if not os.path.isfile(WORD_EMBEDDING_MODEL_FILENAME):
            print("Downloading word-embedding model....")
            urllib.request.urlretrieve(WORD_EMBEDDING_MODEL_URL, WORD_EMBEDDING_MODEL_FILENAME)

    def remove_word_embedding_model(self):
        if os.path.isfile(WORD_EMBEDDING_MODEL_FILENAME):
            print("Removing word-embedding model")
            os.remove(WORD_EMBEDDING_MODEL_FILENAME)

    def assert_wmd_distance(self, tokens1, tokens2, expected_distance):
        actual_distance = word_mover_distance(tokens1, tokens2, self.w2v_model)
        self.assertAlmostEqual(actual_distance, expected_distance, delta=1e-3)

    def test_metrics(self):
        tokens1 = ['president', 'speaks']
        tokens2 = ['president', 'talks']
        expected_distance = 0.19936788082122803
        self.assert_wmd_distance(tokens1, tokens2, expected_distance)

        tokens1 = ['fan', 'book']
        tokens2 = ['apple', 'orange']
        expected_distance = 1.8019972145557404
        self.assert_wmd_distance(tokens1, tokens2, expected_distance)


if __name__ == '__main__':
    unittest.main()