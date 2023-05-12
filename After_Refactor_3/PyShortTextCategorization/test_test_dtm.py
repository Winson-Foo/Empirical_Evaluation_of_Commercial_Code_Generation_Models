import unittest
import re

import pandas as pd
import shorttext
from shorttext.utils import stemword, tokenize


# Constants
STOPWORDS = []
DOC_IDS = []
CORPUS = []
TOKEN_PIPELINE = [
    lambda s: re.sub('[^\w\s]', '', s),
    lambda s: re.sub('[\d]', '', s),
    lambda s: s.lower(),
    lambda s: ' '.join([stemword(token) for token in tokenize(s)])
]


class TestDTM(unittest.TestCase):
    def test_inaugural(self):
        # Load test data
        usprez = TestDTM.load_test_data()

        # Preprocess data
        usprez = TestDTM.preprocess_text_data(usprez)

        # Make corpus and DTM
        TestDTM.make_corpus_and_dtm(usprez)

        # Check results
        TestDTM.check_dtm_results()

    @staticmethod
    def load_test_data():
        usprez = shorttext.data.inaugural()
        global DOC_IDS, STOPWORDS

        DOC_IDS = sorted(usprez.keys())
        STOPWORDS = shorttext.utils.stopwords()
        usprez = [' '.join(usprez[docid]) for docid in DOC_IDS]
        usprez_df = pd.DataFrame({'yrprez': DOC_IDS, 'speech': usprez})
        return usprez_df

    @staticmethod
    def preprocess_text_data(usprez_df):
        global TOKEN_PIPELINE

        txtpreprocessor = shorttext.utils.text_preprocessor(TOKEN_PIPELINE)
        usprez_df['speech'] = usprez_df['speech'].apply(txtpreprocessor)

        return usprez_df

    @staticmethod
    def make_corpus_and_dtm(usprez_df):
        global CORPUS, DOC_IDS

        DOC_IDS = list(usprez_df['yrprez'])
        corpus = [[token for token in doc.split() if token not in STOPWORDS] for doc in usprez_df['speech']]
        CORPUS = corpus

        dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=DOC_IDS, tfidf=True)
        return dtm

    @staticmethod
    def check_dtm_results():
        global DOC_IDS, CORPUS

        dtm = shorttext.utils.DocumentTermMatrix(CORPUS, docids=DOC_IDS, tfidf=True)

        assert len(dtm.dictionary) == 5406
        assert dtm.get_token_occurences(stemword('change'))['2009-Obama'] == 0.013801565936022027
        numdocs, numtokens = dtm.dtm.shape
        assert numdocs == 56
        assert numtokens == 5406
        assert dtm.get_total_termfreq('government') == 0.27584786568258396


if __name__ == '__main__':
    unittest.main()