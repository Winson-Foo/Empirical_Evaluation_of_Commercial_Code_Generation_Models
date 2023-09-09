import numpy as np
import pickle
from typing import List, Optional, Dict
from scipy.sparse import dok_matrix
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

class DocumentTermMatrix:
    """ Document-term matrix for a corpus.

    This is a class that handles the document-term matrix (DTM).
    With a given corpus, users can retrieve term frequency, document frequency,
    and total term frequency. Weighing using tf-idf can be applied.
    """
    def __init__(self, corpus: List[List[str]], docids: Optional[List] = None, tfidf: bool = False) -> None:
        """ Initialize the DocumentTermMatrix class with a given corpus.

        If document IDs (docids) are given, they will be stored and output as appropriate.
        Otherwise, the documents are indexed by numbers.

        Users can choose to weigh by tf-idf. The default is not to weigh.

        The corpus has to be a list of lists, with each of the inside list containing all the tokens
        in each document.

        :param corpus: corpus.
        :param docids: list of designated document IDs. (Default: None)
        :param tfidf: whether to weigh using tf-idf. (Default: False)
        """
        self.dictionary = None
        self.dtm = None
        self.docids = None
        self.docid_dict = None
        self.generate_dtm(corpus, docids, tfidf)

    def generate_dtm(self, corpus: List[List[str]], docids: Optional[List] = None, tfidf: bool = False) -> None:
        """ Generate the inside document-term matrix and other peripheral information objects.
        This is run when the class is instantiated.

        :param corpus: corpus.
        :param docids: list of designated document IDs.
        :param tfidf: whether to weigh using tf-idf.
        """
        self.dictionary = Dictionary(corpus)
        self.dtm = dok_matrix((len(corpus), len(self.dictionary)), dtype=np.float)
        bow_corpus = [self.dictionary.doc2bow(doctokens) for doctokens in corpus]
        if tfidf:
            weighted_model = TfidfModel(bow_corpus)
            bow_corpus = weighted_model[bow_corpus]
        if docids is None:
            self.docid_dict = {i: i for i in range(len(corpus))}
            self.docids = list(range(len(corpus)))
        else:
            self.docids = [docid for docid in docids if docid in range(len(corpus))]
            self.docid_dict = {docid: i for i, docid in enumerate(self.docids)}
        for docid in self.docids:
            for tokenid, count in bow_corpus[self.docid_dict[docid]]:
                self.dtm[self.docid_dict[docid], tokenid] = count

    def get_termfreq(self, docid: int, token: str) -> float:
        """ Retrieve the term frequency of a given token in a particular document.

        Given a token and a particular document ID, compute the term frequency for this
        token. If `tfidf` is set to `True` while instantiating the class, it returns the weighted
        term frequency.

        :param docid: document ID
        :param token: term or token
        :return: term frequency or weighted term frequency of the given token in this document (designated by docid)
        """
        return self.dtm[self.docid_dict[docid], self.dictionary.token2id[token]]

    def get_total_termfreq(self, token: str) -> float:
        """ Retrieve the total occurrences of the given token.

        Compute the total occurrences of the term in all documents. If `tfidf` is set to `True`
        while instantiating the class, it returns the sum of weighted term frequency.

        :param token: term or token
        :return: total occurrences of the given token
        """
        return sum(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_doc_frequency(self, token: str) -> int:
        """ Retrieve the document frequency of the given token.

        Compute the document frequency of the given token, i.e., the number of documents
        that this token can be found.

        :param token: term or token
        :return: document frequency of the given token
        """
        return len(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_token_occurrences(self, token: str) -> Dict[int, float]:
        """ Retrieve the term frequencies of a given token in all documents.

        Compute the term frequencies of the given token for all the documents. If `tfidf` is
        set to be `True` while instantiating the class, it returns the weighted term frequencies.

        This method returns a dictionary of term frequencies with the corresponding document IDs
        as the keys.

        :param token: term or token
        :return: a dictionary of term frequencies with the corresponding document IDs as the keys
        """
        return {self.docids[docidx]: count for (docidx, _), count in self.dtm[:, self.dictionary.token2id[token]].items()}

    def get_doc_tokens(self, docid: int) -> Dict[str, float]:
        """ Retrieve the term frequencies of all tokens in the given document.

        Compute the term frequencies of all tokens for the given document. If `tfidf` is
        set to be `True` while instantiating the class, it returns the weighted term frequencies.

        This method returns a dictionary of term frequencies with the tokens as the keys.

        :param docid: document ID
        :return: a dictionary of term frequencies with the tokens as the keys
        """
        return {self.dictionary[tokenid]: count for (_, tokenid), count in self.dtm[self.docid_dict[docid], :].items()}

    def savemodel(self, prefix: str) -> None:
        """ Save the model.

        :param prefix: prefix of the files
        """
        pickle.dump(self.docids, open(prefix+'_docids.pkl', 'wb'))
        self.dictionary.save(prefix+'_dictionary.dict')
        pickle.dump(self.dtm, open(prefix+'_dtm.pkl', 'wb'))

    def loadmodel(self, prefix: str) -> None:
        """ Load the model.

        :param prefix: prefix of the files
        """
        self.docids = pickle.load(open(prefix+'_docids.pkl', 'rb'))
        self.docid_dict = {docid: i for i, docid in enumerate(self.docids)}
        self.dictionary = Dictionary.load(prefix+'_dictionary.dict')
        self.dtm = pickle.load(open(prefix+'_dtm.pkl', 'rb'))

def load_document_term_matrix(filename: str, compact: bool = True) -> DocumentTermMatrix:
    """ Load presaved Document-Term Matrix (DTM).

    Given the file name (if `compact` is `True`) or the prefix (if `compact` is `False`),
    return the document-term matrix.

    :param filename: file name or prefix
    :param compact: whether it is a compact model. (Default: `True`)
    :return: document-term matrix
    """
    dtm = DocumentTermMatrix([[]])
    if compact:
        dtm.load_compact_model(filename)
    else:
        dtm.loadmodel(filename)
    return dtm