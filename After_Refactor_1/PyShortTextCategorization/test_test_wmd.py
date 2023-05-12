import os
import unittest
import urllib

from shorttext.metrics.wasserstein import word_mover_distance
from shorttext.utils import load_word2vec_model

WORD2VEC_MODEL_LINK = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"
WORD2VEC_MODEL_FILENAME = "test_w2v_model.bin"

class TestWMD(unittest.TestCase):
    """
    Test case for the word mover's distance metric
    """

    def download_word2vec_model(self):
        """
        Download the word-embedding model if it does not already exist
        """
        if not os.path.isfile(WORD2VEC_MODEL_FILENAME):
            urllib.request.urlretrieve(WORD2VEC_MODEL_LINK, WORD2VEC_MODEL_FILENAME)

    def remove_word2vec_model(self):
        """
        Remove the word-embedding model if it exists
        """
        if os.path.isfile(WORD2VEC_MODEL_FILENAME):
            os.remove(WORD2VEC_MODEL_FILENAME)

    def setUp(self):
        """
        Set up the test case by downloading the word-embedding model
        """
        self.download_word2vec_model()
        self.w2v_model = load_word2vec_model(WORD2VEC_MODEL_FILENAME, binary=True)

    def tearDown(self):
        """
        Tear down the test case by removing the word-embedding model
        """
        self.remove_word2vec_model()

    def test_word_mover_distance(self):
        """
        Test the word mover's distance metric using generic test cases
        """
        test_cases = [
            {'tokens1': ['president', 'speaks'], 'tokens2': ['president', 'talks'], 'answer': 0.19936788082122803},
            {'tokens1': ['fan', 'book'], 'tokens2': ['apple', 'orange'], 'answer': 1.8019972145557404},
        ]
        for test_case in test_cases:
            wdistance = word_mover_distance(test_case['tokens1'], test_case['tokens2'], self.w2v_model)
            self.assertAlmostEqual(wdistance, test_case['answer'], delta=1e-3)

if __name__ == '__main__':
    unittest.main()