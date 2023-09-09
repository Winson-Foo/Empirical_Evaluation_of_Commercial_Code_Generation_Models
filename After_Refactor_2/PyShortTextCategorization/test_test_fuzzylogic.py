import unittest
import shorttext

class TestFuzzyLogic(unittest.TestCase):
    def setUp(self):
        self.metric = shorttext.metrics.dynprog.damerau_levenshtein

    def test_similarity_when_mistakes_are_made(self):
        self.assertEqual(self.metric('debug', 'deubg'), 1)
        self.assertEqual(self.metric('intrdependence', 'interdpeendencae'), 3)
        self.assertEqual(shorttext.metrics.dynprog.longest_common_prefix('debug', 'debuag'), 4)

    def test_transposition_of_adjacent_characters(self):
        self.assertEqual(self.metric('independent', 'indeepndent'), 1)
        self.assertEqual(self.metric('providence', 'porvidecne'), 2)

    def test_insertion_of_characters(self):
        self.assertEqual(self.metric('algorithm', 'algorithms'), 1)
        self.assertEqual(self.metric('algorithm', 'algoarithmm'), 2)

    def test_deletion_of_characters(self):
        self.assertEqual(self.metric('algorithm', 'algoithm'), 1)
        self.assertEqual(self.metric('algorithm', 'algorith'), 1)
        self.assertEqual(self.metric('algorithm', 'algrihm'), 2)

    def test_strings_are_identical(self):
        self.assertEqual(self.metric('python', 'python'), 0)
        self.assertEqual(self.metric('sosad', 'sosad'), 0)

    def test_jaccard_similarity_score(self):
        self.assertAlmostEqual(shorttext.metrics.dynprog.similarity('diver', 'driver'), 5./6.)

if __name__ == '__main__':
    unittest.main()