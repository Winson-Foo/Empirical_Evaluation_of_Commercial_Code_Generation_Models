import unittest
import re

import pandas as pd
import shorttext
from shorttext.utils import stemword, tokenize


class TestDTM(unittest.TestCase):

    def setUp(self):
        self.us_presidential_speeches = shorttext.data.inaugural()
        self.doc_ids = sorted(self.us_presidential_speeches.keys())

    def _preprocess_text(self, text):
        """Apply a text pre-processing pipeline to the input text"""
        pipeline = [
            lambda s: re.sub('[^\w\s]', '', s),
            lambda s: re.sub('[\d]', '', s),
            lambda s: s.lower(),
            lambda s: ' '.join([stemword(token) for token in tokenize(s)])
        ]
        preprocessor = shorttext.utils.text_preprocessor(pipeline)
        return preprocessor(text)

    def _prepare_data(self):
        """Prepare the test data for DTM creation"""
        speeches = [' '.join(self.us_presidential_speeches[id]) for id in self.doc_ids]
        speech_df = pd.DataFrame({'speech_id': self.doc_ids, 'speech_text': speeches})
        speech_df = speech_df[['speech_id', 'speech_text']]
        corpus = [self._preprocess_text(speech).split(' ') for speech in speech_df['speech_text']]
        return corpus, self.doc_ids

    def _create_dtm(self, corpus, doc_ids):
        """Create a DocumentTermMatrix from the input corpus and document ids"""
        dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=doc_ids, tfidf=True)
        return dtm

    def test_inaugural(self):
        corpus, doc_ids = self._prepare_data()
        dtm = self._create_dtm(corpus, doc_ids)

        # check results
        self.assertEqual(len(dtm.dictionary), 5406)
        self.assertAlmostEqual(dtm.get_token_occurences(stemword('change'))['2009-Obama'], 0.013801565936022027,
                               places=4)
        numdocs, numtokens = dtm.dtm.shape
        self.assertEqual(numdocs, 56)
        self.assertEqual(numtokens, 5406)
        self.assertAlmostEqual(dtm.get_total_termfreq('government'), 0.27584786568258396,
                               places=4)


if __name__ == '__main__':
    unittest.main()