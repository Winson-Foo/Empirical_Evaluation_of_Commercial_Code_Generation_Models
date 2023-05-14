import numpy as np
from scipy.sparse import csc_matrix
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder
from typing import List
from functools import partial

from shorttext.utils.misc import textfile_generator


class SentenceToCharVecEncoder:
    """ A class that facilitates one-hot encoding from characters to vectors. """
    
    def __init__(self, dictionary: Dictionary, signalchar: str='\n') -> None:
        self.dictionary = dictionary
        self.signalchar = signalchar
        num_chars = len(self.dictionary)
        self.onehot_encoder = OneHotEncoder()
        self.onehot_encoder.fit(np.arange(num_chars).reshape((num_chars, 1)))

    def calculate_preliminary_vector(self, sentence: str) -> np.ndarray:
        return self.onehot_encoder.transform(
            np.array([self.dictionary.token2id[c] for c in sentence]).reshape((len(sentence), 1))
        )

    def encode_sentence(self, 
                        sentence: str, 
                        max_length: int, 
                        add_start_signal: bool=False, 
                        add_end_signal: bool=False
                       ) -> csc_matrix:
        corrected_sentence = (self.signalchar if add_start_signal else '') + sentence[:min(max_length, len(sentence))] + (self.signalchar if add_end_signal else '')
        sentence_vector = self.calculate_preliminary_vector(corrected_sentence).tocsc()
        if sentence_vector.shape[0] == max_length + add_start_signal + add_end_signal:
            return sentence_vector
        else:
            return csc_matrix((sentence_vector.data, sentence_vector.indices, sentence_vector.indptr),
                              shape=(max_length + add_start_signal + add_end_signal, sentence_vector.shape[1]),
                              dtype=np.float64)

    def encode_sentences(self, 
                          sentences: List[str], 
                          max_length: int, 
                          sparse: bool=True, 
                          add_start_signal: bool=False, 
                          add_end_signal: bool=False
                         ) -> List[csc_matrix]:
        encode_fn = partial(self.encode_sentence, 
                            max_length=max_length, 
                            add_start_signal=add_start_signal, 
                            add_end_signal=add_end_signal)
        encoded_sentences = list(map(encode_fn, sentences))
        if sparse:
            return encoded_sentences
        else:
            return np.array([sparse_vec.toarray() for sparse_vec in encoded_sentences])

    def __len__(self) -> int:
        return len(self.dictionary)


def initialize_sentence_to_char_vector_encoder(textfile, encoding=None) -> SentenceToCharVecEncoder:
    dictionary = Dictionary(map(lambda line: [char for char in line], textfile_generator(textfile, encoding=encoding)))
    return SentenceToCharVecEncoder(dictionary)