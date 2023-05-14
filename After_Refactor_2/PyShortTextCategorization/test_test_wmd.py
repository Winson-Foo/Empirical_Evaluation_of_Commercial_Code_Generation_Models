import logging
import os
import unittest
import urllib

from shorttext.metrics.wasserstein import word_mover_distance
from shorttext.utils import load_word2vec_model


class TestWMD(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.info("Downloading word-embedding model...")
        link = "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/test_w2v_model.bin"
        filename = "test_w2v_model.bin"
        if not os.path.isfile(filename):
            urllib.request.urlretrieve(link, filename)
        cls.w2v_model = load_word2vec_model(filename, binary=True)

    @classmethod
    def tearDownClass(cls):
        filename = "test_w2v_model.bin"
        logging.info("Removing word-embedding model...")
        if os.path.isfile(filename):
            os.remove(filename)

    def test_word_mover_distance(self):
        test_cases = [
            {'tokens1': ['president', 'speaks'], 'tokens2': ['president', 'talks'], 'answer': 0.19936788082122803},
            {'tokens1': ['fan', 'book'], 'tokens2': ['apple', 'orange'], 'answer': 1.8019972145557404},
        ]
        for test_case in test_cases:
            with self.subTest(test_case=test_case):
                tokens1 = test_case['tokens1']
                tokens2 = test_case['tokens2']
                answer = test_case['answer']
                wmd = word_mover_distance(tokens1, tokens2, self.w2v_model)
                self.assertAlmostEqual(wmd, answer, delta=1e-3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()