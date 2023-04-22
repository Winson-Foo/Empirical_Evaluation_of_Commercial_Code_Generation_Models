from typing import List, Dict
from multiprocessing import Pool, cpu_count
import math
import nltk
import numpy as np

from nboost.database import DatabaseRow
from nboost.delegates import ResponseDelegate
from nboost.plugins import Plugin

nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(nltk.corpus.stopwords.words('english'))


class BM25:
    """
    Implementation of the BM25 ranking algorithm
    """
    def __init__(self, documents: List[List[str]], tokenizer=None):
        self.num_documents = len(documents)
        self.avg_doc_len = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            documents = self._tokenize_documents(documents)

        word_freqs = self._initialize_documents(documents)
        self._calculate_idf(word_freqs)

    def _tokenize_documents(self, documents: List[List[str]]) -> List[List[str]]:
        """
        Tokenizes the documents using the specified tokenizer
        :param documents: list of documents, where each document is a list of words
        :return: list of tokenized documents
        """
        with Pool(cpu_count()) as pool:
            tokenized_documents = pool.map(self.tokenizer, documents)
        return tokenized_documents

    def _initialize_documents(self, documents: List[List[str]]) -> Dict[str, int]:
        """
        Initializes the document frequency counters and calculates the length of each document
        :param documents: list of documents, where each document is a list of words
        :return: dictionary of words and their document frequencies
        """
        word_freqs = {}  # word -> number of documents with word
        num_words = 0
        for document in documents:
            document_len = len(document)
            self.doc_len.append(document_len)
            num_words += document_len

            word_freq = {}
            for word in document:
                if word not in word_freq:
                    word_freq[word] = 0
                word_freq[word] += 1
            self.doc_freqs.append(word_freq)

            for word, freq in word_freq.items():
                if word not in word_freqs:
                    word_freqs[word] = 0
                word_freqs[word] += 1

        self.avg_doc_len = num_words / self.num_documents
        return word_freqs

    def _calculate_idf(self, word_freqs: Dict[str, int]):
        """
        Calculates the inverse document frequencies for each word in the corpus
        :param word_freqs: dictionary of words and their document frequencies
        :return: None
        """
        idf_sum = 0
        negative_idfs = []
        for word, freq in word_freqs.items():
            idf = math.log((self.num_documents - freq + 0.5) / (freq + 0.5))
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.avg_idf = idf_sum / len(self.idf)

        eps = 0.25 * self.avg_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query: List[str]) -> np.ndarray:
        """
        Calculates the score of each document for the given query
        :param query: list of query words
        :return: array of document scores
        """
        scores = np.zeros(self.num_documents)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            scores += (self.idf.get(q, 0) or 0) * (q_freq * (1.5 + 1) /
                                                   (q_freq + 1.5 * (1 - 0.75 + 0.75 * doc_len / self.avg_doc_len)))
        return scores

    def get_top_n(self, query: List[str], documents: List[str], n=5) -> List[str]:
        """
        Returns the top n documents for the given query
        :param query: list of query words
        :param documents: list of document strings
        :param n: number of documents to return
        :return: list of top n documents
        """
        assert self.num_documents == len(documents), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]


class PreRankPlugin(Plugin):
    """
    A plugin that uses the BM25 algorithm to pre-rank search results based on query relevance
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stemmer = nltk.stem.PorterStemmer()

    def on_response(self, response: ResponseDelegate, db_row: DatabaseRow):
        query = response.request.query
        choices = response.cvalues

        corpus = [self.tokenize(choice) for choice in choices]
        bm25 = BM25(corpus)
        ranks = np.argsort(bm25.get_scores(self.tokenize(query)))[::-1]

        reranked_choices = [response.choices[rank] for rank in ranks]
        response.choices = reranked_choices[:50]

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes the text by removing stopwords and stemming the remaining words
        :param text: string to tokenize
        :return: list of tokenized words
        """
        words = nltk.tokenize.word_tokenize(text.lower())
        words = [self.stemmer.stem(word) for word in words if word not in stop_words]
        return words