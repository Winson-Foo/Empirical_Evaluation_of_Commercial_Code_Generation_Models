import unittest
from urllib.request import urlopen
import shorttext


class SpellCheckTest(unittest.TestCase):
    def setUp(self):
        # Download and decode text file from norvig's website for spell checking purposes
        self.text_input = urlopen('http://norvig.com/big.txt').read()
        self.text_input = self.text_input.decode('utf-8')

    def tearDown(self):
        pass

    def test_norvig_spell_check(self):
        # Create a NorvigSpellCorrector object to train on the downloaded text file
        speller = shorttext.spell.NorvigSpellCorrector()
        speller.train(self.text_input)

        # Validate the spell checker using known misspellings
        self.assertEqual(speller.correct('apple'), 'apple')
        self.assertEqual(speller.correct('appl'), 'apply')


if __name__ == '__main__':
    unittest.main()