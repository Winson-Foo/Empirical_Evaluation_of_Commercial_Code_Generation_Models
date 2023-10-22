from typing import List, Tuple
import numpy as np
import requests
import gensim
from gensim.models import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors
from shorttext.utils import tokenize


def load_word2vec_model(path: str, binary: bool = True) -> KeyedVectors:
    """Load a pre-trained Word2Vec model."""
    return KeyedVectors.load_word2vec_format(path, binary=binary)


def load_fasttext_model(path: str, encoding: str = 'utf-8') -> FastTextKeyedVectors:
    """Load a pre-trained FastText model."""
    return gensim.models.fasttext.load_facebook_vectors(path, encoding=encoding)


def load_poincare_model(path: str, word2vec_format: bool = True, binary: bool = False) -> PoincareKeyedVectors:
    """Load a Poincare embedding model."""
    if word2vec_format:
        return PoincareKeyedVectors.load_word2vec_format(path, binary=binary)
    else:
        return PoincareModel.load(path).kv


def shorttext_to_avgvec(shorttext: str, wvmodel: KeyedVectors) -> np.ndarray:
    """Convert the short text into an averaged embedded vector representation."""
    tokens = tokenize(shorttext)
    vec = np.sum([wvmodel[token] for token in tokens if token in wvmodel], axis=0)
    norm = np.linalg.norm(vec)
    if norm != 0:
        vec /= norm
    return vec


class RESTfulKeyedVectors(KeyedVectors):
    """RESTfulKeyedVectors, for connecting to the API of the preloaded word-embedding vectors loaded by `WordEmbedAPI`."""
    
    def __init__(self, url: str, port: str = '5000'):
        """Initialize the class."""
        self.url = url
        self.port = port

    def _make_request(self, endpoint: str, data: dict) -> dict:
        """Helper function to make requests to the RESTful API."""
        r = requests.post(f"{self.url}:{self.port}/{endpoint}", json=data)
        return r.json()

    def closer_than(self, word1: str, word2: str) -> List[str]:
        """List of words closer than the given distance."""
        return self._make_request("closerthan", {'entity1': word1, 'entity2': word2})

    def distance(self, word1: str, word2: str) -> float:
        """Distance between two words."""
        return self._make_request("distance", {'entity1': word1, 'entity2': word2})['distance']

    def distances(self, word1: str, words: List[str]) -> np.ndarray:
        """List of distances between a word and a list of other words."""
        data = {'entity1': word1, 'other_entities': words}
        return np.array(self._make_request("distances", data)['distances'], dtype=np.float32)

    def get_vector(self, word: str) -> np.ndarray:
        """Word vectors of the given word."""
        data = {'token': word}
        response_dict = self._make_request("get_vector", data)
        if 'vector' in response_dict:
            return np.array(response_dict['vector'])
        else:
            raise KeyError(f"The token {word} does not exist in the model.")

    def most_similar(self, **kwargs) -> List[Tuple[str, float]]:
        """Most similar words."""
        response_list = self._make_request("most_similar", kwargs)
        return [tuple(pair) for pair in response_list]

    def most_similar_to_given(self, word: str, words: List[str]) -> str:
        """Similarity between a word and a list of other words."""
        data = {'entity1': word, 'entities_list': words}
        return self._make_request("most_similar_to_given", data)['token']

    def rank(self, word1: str, word2: str) -> int:
        """Rank."""
        data = {'entity1': word1, 'entity2': word2}
        return self._make_request("rank", data)['rank']

    def save(self, fname_or_handle, **kwargs) -> None:
        """This class does not persist models to a file."""
        raise IOError("The class RESTfulKeyedVectors does not persist models to a file.")

    def similarity(self, word1: str, word2: str) -> float:
        """Similarity between two words."""
        data = {'entity1': word1, 'entity2': word2}
        return self._make_request("similarity", data)['similarity']