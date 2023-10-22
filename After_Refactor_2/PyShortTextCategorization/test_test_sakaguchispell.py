import unittest
import os

from shorttext.spell.sakaguchi import SCRNNSpellCorrector
from shorttext.smartload import smartload_compact_model

class TestSCRNN(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_spell_corrector(self, operation, typo, expected):
        with self.subTest(operation=operation, typo=typo):
            corrector = SCRNNSpellCorrector(operation)
            corrector.train('I am a nerd . Natural language processing is sosad .')

            with open(f'./sosad_{operation}_sakaguchi.bin', 'wb') as f:
                corrector.dump_model(f)

            corrector2 = smartload_compact_model(f'./sosad_{operation}_sakaguchi.bin', None)
            self.assertEqual(corrector2.correct(typo), expected)

            os.remove(f'./sosad_{operation}_sakaguchi.bin')

    def test_NOISE_INSERT(self):
        self.test_spell_corrector('NOISE-INSERT', 'langudge', 'language')

    def test_NOISE_DELETE(self):
        self.test_spell_corrector('NOISE-DELETE', 'langudge', 'language')

    def test_NOISE_REPLACE(self):
        self.test_spell_corrector('NOISE-REPLACE', 'procsesing', 'processing')

    def test_JUMBLE(self):
        operations = ['JUMBLE-WHOLE', 'JUMBLE-BEG', 'JUMBLE-END', 'JUMBLE-INT']
        typos = ['procesisng', 'rocessing', 'processin', 'procesing']
        expected = ['processing', 'processing', 'processing', 'processing']

        for operation, typo, expected in zip(operations, typos, expected):
            self.test_spell_corrector(operation, typo, expected)

if __name__ == '__main__':
    unittest.main()