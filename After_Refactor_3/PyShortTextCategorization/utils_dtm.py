import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from scipy.sparse import dok_matrix
import pickle
from .compactmodel_io import CompactIOMachine
from .classification_exceptions import NotImplementedException

DTM_SUFFIXES = ['_docids.pkl', '_dictionary.dict', '_dtm.pkl']

class DocumentTermMatrix(CompactIOMachine):
    def __init__(self, corpus, docids=[], tfidf=False):
        CompactIOMachine.__init__(self, {'classifier': 'dtm'}, 'dtm', DTM_SUFFIXES)
        if not docids:
            self.docid_dict = {i: i for i in range(len(corpus))}
            self.docids = list(range(len(corpus)))
        else:
            if len(docids) == len(corpus):
                self.docid_dict = {docid: i for i, docid in enumerate(docids)}
                self.docids = docids
            elif len(docids) > len(corpus):
                self.docid_dict = {docid: i for i, docid in zip(range(len(corpus)), docids[:len(corpus)])}
                self.docids = docids[:len(corpus)]
            else:
                self.docid_dict = {docid: i for i, docid in enumerate(docids)}
                self.docid_dict = {i: i for i in range(len(docids), len(corpus))}
                self.docids = docids + list(range(len(docids), len(corpus)))
        self.generate_dtm(corpus, tfidf=tfidf)

    def generate_dtm(self, corpus, tfidf=False):
        self.dictionary = Dictionary(corpus)
        self.dtm = dok_matrix((len(corpus), len(self.dictionary)), dtype=np.float)
        bow_corpus = [self.dictionary.doc2bow(doctokens) for doctokens in corpus]
        if tfidf:
            weighted_model = TfidfModel(bow_corpus)
            bow_corpus = weighted_model[bow_corpus]
        for docid, docidx in self.docid_dict.items():
            for tokenid, count in bow_corpus[docidx]:
                self.dtm[docidx, tokenid] = count

    def get_termfreq(self, docid, token):
        return self.dtm[self.docid_dict[docid], self.dictionary.token2id[token]]

    def get_total_termfreq(self, token):
        return sum(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_doc_frequency(self, token):
        return len(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_token_occurrences(self, token):
        return {
            self.docids[docidx]: count
            for (docidx, _), count in self.dtm[:, self.dictionary.token2id[token]].items()
        }

    def get_doc_tokens(self, docid):
        return {
            self.dictionary[tokenid]: count
            for (_, tokenid), count in self.dtm[self.docid_dict[docid], :].items()
        }

    def generate_dtm_dataframe(self):
        raise NotImplementedException()

    def save_model(self, prefix):
        pickle.dump(self.docids, open(prefix+'_docids.pkl', 'wb'))
        self.dictionary.save(prefix+'_dictionary.dict')
        pickle.dump(self.dtm, open(prefix+'_dtm.pkl', 'wb'))

    def load_model(self, prefix):
        self.docids = pickle.load(open(prefix+'_docids.pkl', 'rb'))
        self.docid_dict = {docid: i for i, docid in enumerate(self.docids)}
        self.dictionary = Dictionary.load(prefix+'_dictionary.dict')
        self.dtm = pickle.load(open(prefix+'_dtm.pkl', 'rb'))


def load_document_term_matrix(filename, compact=True):
    dtm = DocumentTermMatrix([[]])
    if compact:
        dtm.load_compact_model(filename)
    else:
        dtm.load_model(filename)
    return dtm