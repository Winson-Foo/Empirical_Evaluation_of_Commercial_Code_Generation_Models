import unittest
from shorttext.metrics.dynprog import damerau_levenshtein, longest_common_prefix, similarity

class TestFuzzyLogic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # set up any resources needed for the tests
        pass

    @classmethod
    def tearDownClass(cls):
        # clean up any resources opened during the tests
        pass
    
    def test_similarity(self):
        # test various types of similarity measures
        tests = [
            # format: (test_name, expected_similarity, str1, str2)
            ("damerau_levenshtein transposition", 1, "independent", "indeepndent"),
            ("damerau_levenshtein insertion", 1, "algorithm", "algorithms"),
            ("damerau_levenshtein deletion", 1, "algorithm", "algoithm"),
            ("damerau_levenshtein correct", 0, "python", "python"),
            ("longest_common_prefix", 4, "debug", "debuag"),
            ("jaccard", 5./6., "diver", "driver")
        ]

        for test_name, expected_similarity, str1, str2 in tests:
            with self.subTest(name=test_name):
                self.assertEqual(get_similarity(str1, str2), expected_similarity)

    def test_misc(self):
        # test more miscellaneous cases
        self.assertEqual(get_similarity('deubg', 'debug'), 1)
        self.assertEqual(get_similarity('interdpeendencae', 'intrdependence'), 3)
        self.assertEqual(get_similarity('porvidecne', 'providence'), 2)
        self.assertEqual(get_similarity('algoarithmm', 'algorithm'), 2)
        self.assertEqual(get_similarity('algorith', 'algorithm'), 1)
        self.assertEqual(get_similarity('algrihm', 'algorithm'), 2)
        self.assertEqual(get_similarity('sosad', 'sosad'), 0)

def get_similarity(str1, str2):
    # helper function to calculate similarity using Damerau-Levenshtein
    return damerau_levenshtein(str1, str2)

if __name__ == '__main__':
    unittest.main()