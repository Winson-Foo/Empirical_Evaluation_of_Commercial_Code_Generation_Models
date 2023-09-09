import unittest
import shorttext

class TestPreprocessor(unittest.TestCase):

    def test_standard_preprocessor(self):
        preprocessor = shorttext.utils.standard_text_preprocessor_1()
        self.assertEqual(self._process_text(preprocessor, 'I love you.'), 'love')
        self.assertEqual(self._process_text(preprocessor, 'Natural language processing and text mining on fire.'), 'natur languag process text mine fire')
        self.assertEqual(self._process_text(preprocessor, 'I do not think.'), 'think')

    def test_standard_preprocessor_with_custom_stopwords(self):
        preprocessor = shorttext.utils.standard_text_preprocessor_2()
        self.assertEqual(self._process_text(preprocessor, 'I love you.'), 'love')
        self.assertEqual(self._process_text(preprocessor, 'Natural language processing and text mining on fire.'), 'natur languag process text mine fire')
        self.assertEqual(self._process_text(preprocessor, 'I do not think.'), 'not think')

    def _process_text(self, preprocessor, text):
        return preprocessor(text)

if __name__ == '__main__':
    unittest.main()