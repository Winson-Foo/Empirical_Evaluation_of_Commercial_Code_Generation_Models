import unittest
import shorttext

class TestFuzzyLogic(unittest.TestCase):

    def test_similarity_with_transposition(self):
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('debug', 'deubg'), 1)
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('intrdependence', 'interdpeendencae'), 3)

    def test_similarity_with_insertion(self):
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('algorithm', 'algorithms'), 1)
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('algorithm', 'algoarithmm'), 2)

    def test_similarity_with_deletion(self):
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('algorithm', 'algoithm'), 1)
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('algorithm', 'algorith'), 1)
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('algorithm', 'algrihm'), 2)

    def test_similarity_with_correct_word(self):
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('python', 'python'), 0)
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('sosad', 'sosad'), 0)

    def test_similarity_with_jaccard(self):
        self.assertAlmostEqual(shorttext.metrics.dynprog.similarity('diver', 'driver'), 5./6.)

    def test_longest_common_prefix(self):
        self.assertEqual(shorttext.metrics.dynprog.longest_common_prefix('debug', 'debuag'), 4)

    def test_transposition(self):
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('independent', 'indeepndent'), 1)
        self.assertEqual(shorttext.metrics.dynprog.damerau_levenshtein('providence', 'porvidecne'), 2)

if __name__ == '__main__':
    unittest.main()