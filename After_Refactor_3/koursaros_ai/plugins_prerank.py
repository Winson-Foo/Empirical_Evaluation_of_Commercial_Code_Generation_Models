import math
import multiprocessing
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = len(corpus)
        self.tokenizer = tokenizer or (lambda x: x)
        self.doc_len = [len(doc) for doc in corpus]
        self.doc_freqs = [self.get_doc_freqs(doc) for doc in corpus]

    def get_doc_freqs(self, document):
        freqs = {}
        for token in self.tokenizer(document):
            if token not in freqs:
                freqs[token] = 0
            freqs[token] += 1
        return freqs

    def _calculate_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"
        scores = self.get_scores(query)
        top_n = np.argsort(scores)[::-1][:n]
        return [documents[i] for i in top_n]

    @staticmethod
    def tokenize_corpus(corpus):
        with multiprocessing.Pool() as pool:
            return pool.map(tokenize, corpus)

class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        super().__init__(corpus, tokenizer)
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self._calculate_idf()

    def _calculate_idf(self):
        nd = {}
        for freqs in self.doc_freqs:
            for word, freq in freqs.items():
                if word not in nd:
                    nd[word] = 0
                nd[word] += 1

        idf_sum = 0
        neg_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                neg_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in neg_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for token in self.tokenizer(query):
            q_freq = np.array([doc_freqs.get(token, 0) for doc_freqs in self.doc_freqs])
            if token in self.idf:
                idf = self.idf[token]
            else:
                idf = self.epsilon * self.average_idf
            scores += idf * (q_freq * (self.k1 + 1) / (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))

        return scores

def tokenize(paragraph):
    ps = PorterStemmer()
    words = [ps.stem(word) for word in nltk.word_tokenize(paragraph)]
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return filtered_words

class PrerankPlugin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bm25 = None

    def on_response(self, response, db_row):
        query = response.request.query
        choices = response.cvalues
        corpus = BM25.tokenize_corpus(choices)
        if self.bm25 is None or self.bm25.corpus_size != len(corpus):
            self.bm25 = BM25Okapi(corpus)
        scores = self.bm25.get_scores(query)
        indices = np.argsort(scores)[::-1]
        response.choices = [choices[i] for i in indices][:50]