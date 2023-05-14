import unittest
import os

from shorttext.spell.sakaguchi import Sakaguchi
from shorttext.smartload import smartload_compact_model

MODEL_DIR = "./models/"

class TestSakaguchiSpellCorrector(unittest.TestCase):
    def setUp(self):
        self.model = Sakaguchi()

    def tearDown(self):
        del self.model

    def test_noise_insert(self):
        self._test_spell_corrector('NOISE-INSERT')

    def test_noise_delete(self):
        self._test_spell_corrector('NOISE-DELETE')

    def test_noise_replace(self):
        self._test_spell_corrector('NOISE-REPLACE', typo='procsesing', recommendation='processing')

    def test_jumble_whole(self):
        self._test_spell_corrector('JUMBLE-WHOLE')

    def test_jumble_beg(self):
        self._test_spell_corrector('JUMBLE-BEG')

    def test_jumble_end(self):
        self._test_spell_corrector('JUMBLE-END')

    def test_jumble_int(self):
        self._test_spell_corrector('JUMBLE-INT')

    def _test_spell_corrector(self, operation, typo='langudge', recommendation='language'):
        model_path = os.path.join(MODEL_DIR, operation+'.bin')
        self.model.train('I am a nerd . Natural language processing is sosad .')
        self.model.save_compact_model(model_path)

        loaded_model = smartload_compact_model(model_path, None)
        self.assertEqual(self.model.correct(typo), loaded_model.correct(typo))

        print('typo:', typo, '  recommendation:', self.model.correct(typo), '(', recommendation, ')')

        os.remove(model_path)

if __name__ == '__main__':
    unittest.main()