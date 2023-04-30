from nboost.plugins import Plugin
from nboost.delegates import ResponseDelegate
from nboost.database import DatabaseRow
import numpy as np
from multiprocessing import cpu_count, Pool
import math
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')

class BM25:
    def __init__(self, corpus: list, tokenizer=None):
        """
        This class implements the BM25 ranking algorithm.

        :param corpus: The list of documents in the corpus.
        :param tokenizer: The function used to tokenize the documents.
        """
        self.corpus_size = len(corpus)
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus: list) -> dict:
        """
        This method initializes the BM25 algorithm.

        :param corpus: The list of documents in the corpus.
        :return: The dictionary containing the word frequency for each document.
        """
        nd = {}
        num_docs = 0
        for doc in corpus:
            self.doc_len.append(len(doc))
            num_docs += len(doc)

            freq = {}
            for word in doc:
                if word not in freq:
                    freq[word] = 0
                freq[word] += 1
            self.doc_freqs.append(freq)

            for word in freq.keys():
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        self.avg_doc_len = num_docs / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus: list) -> list:
        """
        This method tokenizes the corpus using multiprocessing.

        :param corpus: The list of documents in the corpus.
        :return: The list of tokenized documents.
        """
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd: dict) -> None:
        """
        This calculates the inverse document frequency for each word.

        :param nd: The dictionary containing the number of documents with the word.
        """
        idf_sum = 0  # collect idf sum for average idf
        negative_idfs = []  # collect words with negative idfs to set a special epsilon value
        for word, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = 0.25 * self.average_idf  # epsilon value for words with negative idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: list) -> np.ndarray:
        """
        This method gets BM25 scores for the query and documents.

        :param query: The list of query terms.
        :return: The array of BM25 scores.
        """
        scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            scores += (self.idf.get(q) or 0) * (q_freq * (1.5 + 1) / (q_freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / self.avg_doc_len)))
        return scores

    def get_top_n(self, query: list, documents: list, n: int = 5) -> list:
        """
        This method gets the top N documents for the query.

        :param query: The list of query terms.
        :param documents: The list of documents in the corpus.
        :param n: The number of documents to return.
        :return: The top N documents.
        """
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class BM25Okapi(BM25):
    def __init__(self, corpus: list, tokenizer=None, k1: float = 1.5, b: float = 0.75, epsilon: float = 0.25):
        """
        This class implements the Okapi BM25 ranking algorithm.

        :param corpus: The list of documents in the corpus.
        :param tokenizer: The function used to tokenize the documents.
        :param k1: The parameter for scaling term frequency.
        :param b: The parameter for scaling document length.
        :param epsilon: The parameter for minimum idf value.
        """
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd: dict) -> None:
        """
        This calculates the inverse document frequency for each word.
        It sets a minimum idf value to prevent negative idf scores.

        :param nd: The dictionary containing the number of documents with the word.
        """
        idf_sum = 0  # collect idf sum for average idf
        negative_idfs = []  # collect words with negative idfs to set a special epsilon value
        for word, freq in nd.items():
            idf = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf  # epsilon value for words with negative idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: list) -> np.ndarray:
        """
        This method gets Okapi BM25 scores for the query and documents.
        It sets a minimum idf value to prevent negative idf scores.

        :param query: The list of query terms.
        :return: The array of Okapi BM25 scores.
        """
        scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            scores += (self.idf.get(q) or self.epsilon * self.average_idf) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)))
        return scores


class PrerankPlugin(Plugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stemmer = PorterStemmer()

    def on_response(self, response: ResponseDelegate, db_row: DatabaseRow) -> None:
        """
        This method is called when the response is generated.
        It reranks the choices using the Okapi BM25 algorithm.

        :param response: The response delegate.
        :param db_row: The database row.
        """
        query = response.request.query
        choices = response.cvalues

        corpus = [self.tokenize(choice) for choice in choices]
        okapi_bm25 = BM25Okapi(corpus)
        ranks = np.argsort(okapi_bm25.get_scores(self.tokenize(query)))[::-1]

        reranked_choices = [response.choices[rank] for rank in ranks]
        response.choices = reranked_choices[:50]

    def tokenize(self, paragraph: str) -> list:
        """
        This tokenizes the paragraph and returns the list of stemmed and filtered words.

        :param paragraph: The text to tokenize.
        :return: The list of filtered and stemmed words.
        """
        words = [self.stemmer.stem(word) for word in word_tokenize(paragraph)]
        filtered_words = [word for word in words if word not in stopwords.words('english')]
        return filtered_words```