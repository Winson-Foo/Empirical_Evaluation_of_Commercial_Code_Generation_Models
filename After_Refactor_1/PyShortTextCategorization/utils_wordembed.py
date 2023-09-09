import numpy as np
import gensim
import requests

from shorttext.utils import tokenize

def load_word2vec_model(path, binary=True):
    """Load a pre-trained Word2Vec model."""
    return gensim.models.KeyedVectors.load_word2vec_format(path, binary=binary)

def load_fasttext_model(path, encoding="utf-8"):
    """Load a pre-trained FastText model."""
    return gensim.models.fasttext.load_facebook_vectors(path, encoding=encoding)

def load_poincare_model(path, word2vec_format=True, binary=False):
    """Load a Poincare embedding model."""
    if word2vec_format:
        return gensim.models.poincare.PoincareKeyedVectors.load_word2vec_format(path, binary=binary)
    else:
        return gensim.models.poincare.PoincareModel.load(path).kv

def short_text_to_avg_embedded_vector(text, model):
    """Convert short text into an averaged embedded vector representation."""
    vec = np.sum([model[token] for token in tokenize(text) if token in model], axis=0)

    # normalize the resulting vector
    norm = np.linalg.norm(vec)
    if norm != 0:
        vec /= norm

    return vec

class RESTfulKeyedVectors(gensim.models.KeyedVectors):
    """Class for connecting to the API of the preloaded word-embedding vectors."""
    def __init__(self, url, port="5000"):
        """Initialize the class."""
        self.url = url
        self.port = port

    def closer_than(self, word1, word2):
        """List all words closer than given word."""
        r = requests.post(self.url + ":" + self.port + "/closerthan",
                          json={"entity1": word1, "entity2": word2})
        return r.json()

    def distance(self, word1, word2):
        """Calculate the distance between two words."""
        r = requests.post(self.url + ":" + self.port + "/distance",
                          json={"entity1": word1, "entity2": word2})
        return r.json()["distance"]

    def distances(self, word, other_words=()):
        """Calculate distances between one word and other words."""
        r = requests.post(self.url + ":" + self.port + "/distances",
                          json={"entity1": word, "other_entities": other_words})
        return np.array(r.json()["distances"], dtype=np.float32)

    def get_vector(self, word):
        """Get vector of the given word."""
        r = requests.post(self.url + ":" + self.port + "/get_vector", json={"token": word})
        returned_dict = r.json()
        if "vector" in returned_dict:
            return np.array(returned_dict["vector"])
        else:
            raise KeyError("The token {} does not exist in the model.".format(word))

    def most_similar(self, **kwargs):
        """Find the words most similar to the given word."""
        r = requests.post(self.url + ":" + self.port + "/most_similar", json=kwargs)
        return [tuple(pair) for pair in r.json()]

    def most_similar_to_given(self, word, words_list):
        """Find the word in the words list that is most similar to the given word."""
        r = requests.post(self.url + ":" + self.port + "/most_similar_to_given",
                          json={"entity1": word, "entities_list": words_list})
        return r.json()["token"]

    def rank(self, word1, word2):
        """Calculate the rank of the given word."""
        r = requests.post(self.url + ":" + self.port + "/rank",
                          json={"entity1": word1, "entity2": word2})
        return r.json()["rank"]

    def save(self, fname_or_handle, **kwargs):
        """Disapprove model saving in RESTfulKeyedVectors."""
        raise IOError("The class RESTfulKeyedVectors does not persist models to a file.")

    def similarity(self, word1, word2):
        """Calculate the similarity between two words."""
        r = requests.post(self.url + ":" + self.port + "/similarity",
                          json={"entity1": word1, "entity2": word2})
        return r.json()["similarity"]