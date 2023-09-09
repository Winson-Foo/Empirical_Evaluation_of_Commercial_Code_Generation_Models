from functools import partial
from typing import List, Optional, Union

import numpy as np
from scipy.sparse import csc_matrix
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder

from shorttext.utils.misc import textfile_generator


class SentenceToCharVecEncoder:
    """ A class that facilitates one-hot encoding from characters to vectors. """

    def __init__(self, dictionary: Dictionary, signal_char: str = '\n'):
        """
        Initialize the one-hot encoding class.

        :param dictionary: a gensim dictionary
        :param signal_char: signal character, useful for seq2seq models (Default: '\n')
        """
        self.dictionary = dictionary
        self.signal_char = signal_char
        num_chars = len(self.dictionary)
        self.onehot_encoder = OneHotEncoder()
        self.onehot_encoder.fit(np.arange(num_chars).reshape((num_chars, 1)))

    def calculate_prelim_vec(self, sentence: str) -> np.array:
        """
        Convert the sentence to a one-hot vector.

        :param sentence: sentence
        :return: a one-hot vector, with each element the code of that character
        """
        return self.onehot_encoder.transform(
            np.array([self.dictionary.token2id[char] for char in sentence]).reshape((len(sentence), 1))
        ).toarray()

    def encode_sentence(
            self,
            sentence: str,
            max_len: int,
            start_sig: bool = False,
            end_sig: bool = False
    ) -> np.array:
        """
        Encode one sentence to a sparse matrix, with each row the expanded vector of each character.

        :param sentence: sentence
        :param max_len: maximum length of the sentence
        :param start_sig: signal character at the beginning of the sentence (Default: False)
        :param end_sig: signal character at the end of the sentence (Default: False)
        :return: matrix representing the sentence
        """
        cor_sent = (self.signal_char if start_sig else '') + sentence[:min(max_len, len(sentence))] \
                   + (self.signal_char if end_sig else '')
        sent_vec = self.calculate_prelim_vec(cor_sent)
        if sent_vec.shape[0] == max_len + start_sig + end_sig:
            return sent_vec
        else:
            return np.pad(sent_vec, ((0, max_len + start_sig + end_sig - sent_vec.shape[0]), (0, 0)),
                          mode='constant')

    def encode_sentences(
            self,
            sentences: List[str],
            max_len: int,
            sparse: bool = True,
            start_sig: bool = False,
            end_sig: bool = False
    ) -> Union[List[csc_matrix], np.array]:
        """
        Encode many sentences into a rank-3 tensor.

        :param sentences: sentences
        :param max_len: maximum length of one sentence
        :param sparse: whether to return a sparse matrix (Default: True)
        :param start_sig: signal character at the beginning of the sentence (Default: False)
        :param end_sig: signal character at the end of the sentence (Default: False)
        :return: rank-3 tensor of the sentences
        """
        encode_sent_func = partial(self.encode_sentence, start_sig=start_sig, end_sig=end_sig, max_len=max_len)
        list_encoded_sentences_map = [encode_sent_func(sent) for sent in sentences]
        if sparse:
            return list(map(csc_matrix, list_encoded_sentences_map))
        else:
            return np.array(list_encoded_sentences_map)

    def __len__(self):
        return len(self.dictionary)


def init_sentence_to_char_vec_encoder(text_file, encoding: Optional[str] = None) -> SentenceToCharVecEncoder:
    """
    Instantiate a class of SentenceToCharVecEncoder from a text file.

    :param text_file: text file
    :param encoding: encoding of the text file (Default: None)
    :return: an instance of SentenceToCharVecEncoder
    """
    dictionary = Dictionary(
        [list(line) for line in textfile_generator(text_file, encoding=encoding)]
    )
    return SentenceToCharVecEncoder(dictionary)