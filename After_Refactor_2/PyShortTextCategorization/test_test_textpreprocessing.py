import unittest
from shorttext.utils import standard_text_preprocessor_1, standard_text_preprocessor_2

class TestTextPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocessor1 = standard_text_preprocessor_1()
        cls.preprocessor2 = standard_text_preprocessor_2()
        
    def test_standard_pipeline(self):
        self.assertEqual(self.preprocessor1('I love you.'), 'love')
        self.assertEqual(self.preprocessor1('Natural language processing and text mining on fire.'), 'natur languag process text mine fire')
        self.assertEqual(self.preprocessor1('I do not think.'), 'think')

    def test_standard_pipeline_different_stopwords(self):
        self.assertEqual(self.preprocessor2('I love you.'), 'love')
        self.assertEqual(self.preprocessor2('Natural language processing and text mining on fire.'), 'natur languag process text mine fire')
        self.assertEqual(self.preprocessor2('I do not think.'), 'not think')

if __name__ == '__main__':
    unittest.main()