import unittest

import shorttext

class TestShortText(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_standard_text_preprocessor_1(self):
        """
        Test that standard_text_preprocessor_1() returns the expected output.
        """
        preprocessor = shorttext.utils.standard_text_preprocessor_1()
        self.assert_preprocessor_output(preprocessor)

    def test_standard_text_preprocessor_2(self):
        """
        Test that standard_text_preprocessor_2() returns the expected output.
        """
        preprocessor = shorttext.utils.standard_text_preprocessor_2()
        self.assert_preprocessor_output(preprocessor)

    def assert_preprocessor_output(self, preprocessor):
        """
        Helper function that checks the output of the preprocessor.
        """
        self.assertEqual(preprocessor('I love you.'), 'love')
        self.assertEqual(preprocessor('Natural language processing and text mining on fire.'), 'natur languag process text mine fire')
        self.assertEqual(preprocessor('I do not think.'), 'think')

if __name__ == '__main__':
    unittest.main()