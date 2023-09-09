import unittest
import os

from shorttext.spell.sakaguchi import SCRNNSpellCorrector
from shorttext.smartload import smartload_compact_model

class TestSCRNN(unittest.TestCase):
    def setUp(self):
        self.corrector = SCRNNSpellCorrector('NOISE-INSERT')
        self.corrector.train('I am a nerd. Natural language processing is sosad.')
        self.corrector.save_compact_model('./sosad_NOISE-INSERT_sakaguchi.bin')
    
    def tearDown(self):
        self._delete_compact_model_file('./sosad_NOISE-INSERT_sakaguchi.bin')

    def _load_compact_model_file(self, filename):
        corrector = smartload_compact_model(filename, None)
        return corrector

    def _delete_compact_model_file(self, filename):
        os.remove(filename)

    def test_noise_insert(self):
        typo = 'langudge'
        expected_recommendation = 'language'
        corrector2 = self._load_compact_model_file('./sosad_NOISE-INSERT_sakaguchi.bin')

        self.assertEqual(self.corrector.correct(typo), corrector2.correct(typo))
        self.assertEqual(self.corrector.correct(typo), expected_recommendation)

    def test_noise_delete(self):
        typo = 'lanugage'
        expected_recommendation = 'language'
        self.corrector.operation = 'NOISE-DELETE'
        self.corrector.save_compact_model('./sosad_NOISE-DELETE_sakaguchi.bin')
        corrector2 = self._load_compact_model_file('./sosad_NOISE-DELETE_sakaguchi.bin')

        self.assertEqual(self.corrector.correct(typo), corrector2.correct(typo))
        self.assertEqual(self.corrector.correct(typo), expected_recommendation)

        self._delete_compact_model_file('./sosad_NOISE-DELETE_sakaguchi.bin')

    def test_noise_replace(self):
        typo = 'procsesing'
        expected_recommendation = 'processing'
        self.corrector.operation = 'NOISE-REPLACE'
        self.corrector.save_compact_model('./sosad_NOISE-REPLACE_sakaguchi.bin')
        corrector2 = self._load_compact_model_file('./sosad_NOISE-REPLACE_sakaguchi.bin')

        self.assertEqual(self.corrector.correct(typo), corrector2.correct(typo))
        self.assertEqual(self.corrector.correct(typo), expected_recommendation)

        self._delete_compact_model_file('./sosad_NOISE-REPLACE_sakaguchi.bin')

    def test_jumble_whole(self):
        typo = 'ngiurlaap'
        expected_recommendation = 'natural'
        self.corrector.operation = 'JUMBLE-WHOLE'
        self.corrector.save_compact_model('./sosad_JUMBLE-WHOLE_sakaguchi.bin')
        corrector2 = self._load_compact_model_file('./sosad_JUMBLE-WHOLE_sakaguchi.bin')

        self.assertEqual(self.corrector.correct(typo), corrector2.correct(typo))
        self.assertEqual(self.corrector.correct(typo), expected_recommendation)

        self._delete_compact_model_file('./sosad_JUMBLE-WHOLE_sakaguchi.bin')

    def test_jumble_beg(self):
        typo = 'ntraau'
        expected_recommendation = 'natural'
        self.corrector.operation = 'JUMBLE-BEG'
        self.corrector.save_compact_model('./sosad_JUMBLE-BEG_sakaguchi.bin')
        corrector2 = self._load_compact_model_file('./sosad_JUMBLE-BEG_sakaguchi.bin')

        self.assertEqual(self.corrector.correct(typo), corrector2.correct(typo))
        self.assertEqual(self.corrector.correct(typo), expected_recommendation)

        self._delete_compact_model_file('./sosad_JUMBLE-BEG_sakaguchi.bin')

    def test_jumble_end(self):
        typo = 'truali'
        expected_recommendation = 'natural'
        self.corrector.operation = 'JUMBLE-END'
        self.corrector.save_compact_model('./sosad_JUMBLE-END_sakaguchi.bin')
        corrector2 = self._load_compact_model_file('./sosad_JUMBLE-END_sakaguchi.bin')

        self.assertEqual(self.corrector.correct(typo), corrector2.correct(typo))
        self.assertEqual(self.corrector.correct(typo), expected_recommendation)

        self._delete_compact_model_file('./sosad_JUMBLE-END_sakaguchi.bin')

    def test_jumble_int(self):
        typo = 'naaturl'
        expected_recommendation = 'natural'
        self.corrector.operation = 'JUMBLE-INT'
        self.corrector.save_compact_model('./sosad_JUMBLE-INT_sakaguchi.bin')
        corrector2 = self._load_compact_model_file('./sosad_JUMBLE-INT_sakaguchi.bin')

        self.assertEqual(self.corrector.correct(typo), corrector2.correct(typo))
        self.assertEqual(self.corrector.correct(typo), expected_recommendation)

        self._delete_compact_model_file('./sosad_JUMBLE-INT_sakaguchi.bin')


if __name__ == '__main__':
    unittest.main()