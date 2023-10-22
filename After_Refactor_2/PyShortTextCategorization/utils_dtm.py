import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from scipy.sparse import dok_matrix
import pandas as pd
import pickle


class DocumentTermMatrix:
    """ Document-term matrix for corpus.

    This is a class that handles the document-term matrix (DTM). With a given corpus, users can
    retrieve term frequency, document frequency, and total term frequency. Weighing using tf-idf
    can be applied.
    """

    def __init__(self, corpus, docids=None, tfidf=False):
        """ Initialize the document-term matrix (DTM) class with a given corpus.

        If document IDs (docids) are given, it will be stored and output as approrpriate.
        If not, the documents are indexed by numbers.

        Users can choose to weigh by tf-idf. The default is not to weigh.

        The corpus has to be a list of lists, with each of the inside list contains all the tokens
        in each document.

        :param corpus: corpus.
        :param docids: list of designated document IDs. (Default: None)
        :param tfidf: whether to weigh using tf-idf. (Default: False)
        :type corpus: list
        :type docids: list
        :type tfidf: bool
        """
        self.dictionary = Dictionary(corpus)

        if docids is None:
            self.docids = range(len(corpus))
        else:
            self.docids = docids

        self.docid_dict = {docid: i for i, docid in enumerate(self.docids)}

        self.dtm = dok_matrix((len(corpus), len(self.dictionary)), dtype=np.float)
        bow_corpus = [self.dictionary.doc2bow(doctokens) for doctokens in corpus]

        if tfidf:
            weighted_model = TfidfModel(bow_corpus)
            bow_corpus = weighted_model[bow_corpus]

        for docid in self.docids:
            for tokenid, count in bow_corpus[self.docid_dict[docid]]:
                self.dtm[self.docid_dict[docid], tokenid] = count

    def get_termfreq(self, docid, token):
        """ Retrieve the term frequency of a given token in a particular document.

        Given a token and a particular document ID, compute the term frequency for this
        token. If `tfidf` is set to `True` while instantiating the class, it returns the weighted
        term frequency.

        :param docid: document ID
        :param token: term or token
        :return: term frequency or weighted term frequency of the given token in this document (designated by docid)
        :type docid: any
        :type token: str
        :rtype: numpy.float
        """
        return self.dtm[self.docid_dict[docid], self.dictionary.token2id[token]]

    def get_total_termfreq(self, token):
        """ Retrieve the total occurrences of the given token.

        Compute the total occurrences of the term in all documents. If `tfidf` is set to `True`
        while instantiating the class, it returns the sum of weighted term frequency.

        :param token: term or token
        :return: total occurrences of the given token
        :type token: str
        :rtype: numpy.float
        """
        return sum(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_doc_frequency(self, token):
        """ Retrieve the document frequency of the given token.

        Compute the document frequency of the given token, i.e., the number of documents
        that this token can be found.

        :param token: term or token
        :return: document frequency of the given token
        :type token: str
        :rtype: int
        """
        return len(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_token_occurrences(self, token):
        """ Retrieve the term frequencies of a given token in all documents.

        Compute the term frequencies of the given token for all the documents. If `tfidf` is
        set to be `True` while instantiating the class, it returns the weighted term frequencies.

        This method returns a dictionary of term frequencies with the corresponding document IDs
        as the keys.

        :param token: term or token
        :return: a dictionary of term frequencies with the corresponding document IDs as the keys
        :type token: str
        :rtype: dict
        """
        return {self.docids[docidx]: count for (docidx, _), count in self.dtm[:, self.dictionary.token2id[token]].items()}

    def get_doc_tokens(self, docid):
        """ Retrieve the term frequencies of all tokens in the given document.

        Compute the term frequencies of all tokens for the given document. If `tfidf` is
        set to be `True` while instantiating the class, it returns the weighted term frequencies.

        This method returns a dictionary of term frequencies with the tokens as the keys.

        :param docid: document ID
        :return: a dictionary of term frequencies with the tokens as the keys
        :type docid: any
        :rtype: dict
        """
        return {self.dictionary[tokenid]: count for (_, tokenid), count in self.dtm[self.docid_dict[docid], :].items()}

    def generate_dtm_dataframe(self, topN=1000):
        """ Generate the data frame of the document-term matrix.

        :param topN: number of top tokens.
        :return: data frame of the document-term matrix
        :rtype: pandas.DataFrame
        """
        res_rows = []
        res_cols = ['docid']
        for t in self.dictionary.token2id:
            res_cols.append(t)
            res_dict = {self.docids[docidx]: count for (docidx, _), count in self.dtm[:, self.dictionary.token2id[t]].items()}
            td = pd.DataFrame([[docid, res_dict[docid] if docid in res_dict else 0] for docid in self.docids], columns=['docid', t])
            td.set_index('docid', inplace=True)
            res_rows.append(td)
        res = pd.concat(res_rows, axis=1)
        res.columns = res_cols
        res.fillna(0, inplace=True)

        top_tokens = res.loc[:, res.columns != 'docid'].sum(axis=0).sort_values(ascending=False).iloc[:topN].index
        res = res.loc[:, ['docid']+list(top_tokens)]
        return res

    def savemodel(self, prefix):
        """ Save the model.

        :param prefix: prefix of the files
        :return: None
        :type prefix: str
        """
        model_dict = {'docids': self.docids, 'docid_dict': self.docid_dict, 'dictionary': self.dictionary, 'dtm': self.dtm}
        with open(f'{prefix}.pkl', 'wb') as f:
            pickle.dump(model_dict, f)

    def loadmodel(self, prefix):
        """ Load the model.

        :param prefix: prefix of the files
        :return: None
        :type prefix: str
        """
        with open(f'{prefix}.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            self.docids = model_dict['docids']
            self.docid_dict = model_dict['docid_dict']
            self.dictionary = model_dict['dictionary']
            self.dtm = model_dict['dtm']


def load_DocumentTermMatrix(filename):
    """ Load presaved Document-Term Matrix (DTM).

    Given the file name, returns the document-term matrix.

    :param filename: file name
    :return: document-term matrix
    :type filename: str
    :rtype: DocumentTermMatrix
    """
    with open(f'{filename}.pkl', 'rb') as f:
        model_dict = pickle.load(f)
        dtm = DocumentTermMatrix([[]])
        dtm.docids = model_dict['docids']
        dtm.docid_dict = model_dict['docid_dict']
        dtm.dictionary = model_dict['dictionary']
        dtm.dtm = model_dict['dtm']
        return dtm