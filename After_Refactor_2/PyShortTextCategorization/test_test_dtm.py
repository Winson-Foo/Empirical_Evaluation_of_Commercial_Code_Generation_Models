import unittest
import re

import pandas as pd
import shorttext
from shorttext.utils import stemword, tokenize, text_preprocessor, DocumentTermMatrix


def prepare_data(data):
    docids = sorted(data.keys())
    speeches = [' '.join(data[docid]) for docid in docids]
    return pd.DataFrame({'yrprez': docids, 'speech': speeches})[['yrprez', 'speech']]


def preprocess_text(pipeline, text):
    return ' '.join([stemword(token) for token in tokenize(pipeline(text))])


def make_corpus(data, pipeline):
    return [preprocess_text(pipeline, speech).split(' ') for speech in data['speech']]


class TestDTM(unittest.TestCase):
    def test_inaugural(self):
        # preparing data
        data = shorttext.data.inaugural()
        data_df = prepare_data(data)

        # preprocesser defined
        pipeline = text_preprocessor([
            lambda s: re.sub('[^\w\s]', '', s),
            lambda s: re.sub('[\d]', '', s),
            lambda s: s.lower(),
        ])

        # corpus making
        corpus = make_corpus(data_df, pipeline)

        # making DTM
        dtm = DocumentTermMatrix(corpus, docids=data_df['yrprez'], tfidf=True)

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