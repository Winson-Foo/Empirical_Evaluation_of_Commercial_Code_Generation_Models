import unittest
from urllib.request import urlopen
import shorttext

BIG_TEXT_URL = 'http://norvig.com/big.txt'

class TestSpellCheck(unittest.TestCase):
    def setUp(self):
        self.big_text = urlopen(BIG_TEXT_URL).read().decode('utf-8')
        self.speller = shorttext.spell.NorvigSpellCorrector()
        self.speller.train(self.big_text)

    def tearDown(self):
        pass

    def test_correct_spelling(self):
        self.assertEqual(self.speller.correct('apple'), 'apple')
    
    def test_apply_spelling(self):
        self.assertEqual(self.speller.correct('appl'), 'apply')

if __name__ == '__main__':
    unittest.main()