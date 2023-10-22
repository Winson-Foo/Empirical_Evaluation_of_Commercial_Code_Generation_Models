import unittest
import sys
from contextlib import closing
from urllib.request import urlopen

import shorttext


def get_test_data():
    # Retrieve test data from the URL and decode it.
    with closing(urlopen('http://norvig.com/big.txt')) as f:
        text = f.read().decode('utf-8')
    return text


class TestSpellCheck(unittest.TestCase):
    def setUp(self):
        # Retrieve test data once before each test method.
        self.text = get_test_data()

    def tearDown(self):
        pass

    def test_norvig(self):
        # Train the spell checker with the test data.
        speller = shorttext.spell.NorvigSpellCorrector()
        speller.train(self.text)

        # Test the accuracy of the spell checker.
        self.assertEqual(speller.correct('apple'), 'apple')
        self.assertEqual(speller.correct('appl'), 'apply')


if __name__ == '__main__':
    unittest.main()