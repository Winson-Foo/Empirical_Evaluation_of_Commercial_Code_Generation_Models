from typing import List
import numpy as np
from scipy.sparse import csc_matrix
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder

from shorttext.utils.misc import textfile_generator


class CharVectorizer:
    """ One-hot encoder from characters to vectors. """
    def __init__(self, dictionary: Dictionary):
        """ Initialize the one-hot encoding class.

        :param dictionary: a gensim dictionary
        """
        self.dictionary = dictionary
        self.onehot_encoder = OneHotEncoder(categories=[range(len(dictionary))])

    def vectorize(self, text: str) -> np.ndarray:
        """ Convert the text to a one-hot vector.

        :param text: text
        :return: a one-hot vector, with each element the code of that character
        """
        ids = [self.dictionary.token2id[c] for c in text]
        return self.onehot_encoder.fit_transform(np.array(ids).reshape(-1, 1)).toarray()


class SentenceVectorizer:
    """ One-hot encoder from sentences to matrices. """
    def __init__(self, char_vectorizer: CharVectorizer, maxlen: int, signalchar: str = '\n'):
        """ Initialize the sentence vectorizer.

        :param char_vectorizer: character vectorizer
        :param maxlen: maximum length of the sentence
        :param signalchar: signal character, useful for seq2seq models (Default: '\n')
        """
        self.char_vectorizer = char_vectorizer
        self.maxlen = maxlen
        self.signalchar = signalchar

    def vectorize(self, sentence: str, startsig: bool = False, endsig: bool = False) -> csc_matrix:
        """ Encode one sentence to a sparse matrix, with each row the expanded vector of each character.

        :param sentence: sentence
        :param startsig: signal character at the beginning of the sentence (Default: False)
        :param endsig: signal character at the end of the sentence (Default: False)
        :return: matrix representing the sentence
        """
        cor_sent = (self.signalchar if startsig else '') + sentence[:min(self.maxlen, len(sentence))] + (self.signalchar if endsig else '')
        sentence_vec = self.char_vectorizer.vectorize(cor_sent)
        if sentence_vec.shape[0] == self.maxlen + startsig + endsig:
            return csc_matrix(sentence_vec)
        else:
            return csc_matrix((sentence_vec.data, sentence_vec.indices, sentence_vec.indptr),
                              shape=(self.maxlen + startsig + endsig, sentence_vec.shape[1]),
                              dtype=np.float64)


class CorpusVectorizer:
    """ One-hot encoder from a corpus of sentences to a rank-3 tensor. """
    def __init__(self, sentence_vectorizer: SentenceVectorizer):
        """ Initialize the corpus vectorizer.

        :param sentence_vectorizer: sentence vectorizer
        """
        self.sentence_vectorizer = sentence_vectorizer

    def vectorize(self, sentences: List[str], sparse: bool = True, startsig: bool = False, endsig: bool = False) -> np.ndarray:
        """ Encode many sentences into a rank-3 tensor.

        :param sentences: sentences
        :param sparse: whether to return a sparse matrix (Default: True)
        :param startsig: signal character at the beginning of the sentence (Default: False)
        :param endsig: signal character at the end of the sentence (Default: False)
        :return: rank-3 tensor of the sentences
        """
        sentence_vectors = [self.sentence_vectorizer.vectorize(s, startsig=startsig, endsig=endsig) for s in sentences]
        if sparse:
            return sentence_vectors
        else:
            return np.array([sparsevec.toarray() for sparsevec in sentence_vectors])


def init_corpus_vectorizer(textfile, encoding=None, maxlen=100):
    """ Instantiate a class of CorpusVectorizer from a text file.

    :param textfile: text file
    :param encoding: encoding of the text file (Default: None)
    :param maxlen: maximum length of one sentence
    :return: an instance of CorpusVectorizer
    """
    char_vectorizer = CharVectorizer(Dictionary(map(list, textfile_generator(textfile, encoding=encoding))))
    sentence_vectorizer = SentenceVectorizer(char_vectorizer, maxlen=maxlen)
    return CorpusVectorizer(sentence_vectorizer)