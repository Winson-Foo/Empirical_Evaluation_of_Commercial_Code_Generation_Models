import pickle
from collections import defaultdict
from typing import Callable, Dict

import numpy as np
from scipy.spatial.distance import cosine
from gensim.models.keyedvectors import KeyedVectors

from shorttext.utils.classification_exceptions import ModelNotTrainedException
from shorttext.utils import shorttext_to_avgvec
from shorttext.utils.compactmodel_io import CompactIOMachine


class SumEmbeddedVecClassifier(CompactIOMachine):
    """
    This is a supervised classification algorithm for short text categorization.
    Each class label has a few short sentences, where each token is converted
    to an embedded vector, given by a pre-trained word-embedding model (e.g., Google Word2Vec model).
    They are then summed up and normalized to a unit vector for that particular class labels.
    To perform prediction, the input short sentences is converted to a unit vector
    in the same way. The similarity score is calculated by the cosine similarity.

    A pre-trained Google Word2Vec model can be downloaded `here
    <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit>`_.
    """

    def __init__(self, wv_model: KeyedVectors, vec_size: int = None, sim_fcn: Callable = cosine):
        """
        Initialize the classifier.

        :param wv_model: Word2Vec model
        :param vec_size: length of the embedded vectors in the model (Default: None, directly extracted from word-embedding model)
        :param sim_fcn: similarity function (Default: cosine similarity)
        """
        CompactIOMachine.__init__(self, {'classifier': 'sumvec'}, 'sumvec', ['_embedvecdict.pkl'])
        self._add_vec = defaultdict(lambda: np.zeros(self.vec_size))
        self.wv_model = wv_model
        self.vec_size = self.wv_model.vector_size if vec_size is None else vec_size
        self.sim_fcn = sim_fcn
        self.trained = False

    def train(self, class_dict: Dict[str, str]):
        """
        Train the classifier.

        If this has not been run, or a model was not loaded by :func:`~load_model`,
        a `ModelNotTrainedException` will be raised while performing prediction or saving
        the model.

        :param class_dict: training data
        """
        for class_type, short_texts in class_dict.items():
            self._add_vec[class_type] = np.sum([self._short_text_to_embed_vec(short_text)
                                             for short_text in short_texts],
                                            axis=0)
            self._add_vec[class_type] /= np.linalg.norm(self._add_vec[class_type])
        self._add_vec = dict(self._add_vec)
        self.trained = True

    def save_model(self, file_prefix: str):
        """
        Save the trained model into files.

        Given the prefix of the file paths, save the model into files, with name given by the prefix,
        and add "_embedvecdict.pickle" at the end. If there is no trained model, a `ModelNotTrainedException`
        will be thrown.

        :param file_prefix: prefix of the file path
        """
        if not self.trained:
            raise ModelNotTrainedException()
        model_file_path = f"{file_prefix}_embedvecdict.pkl"
        pickle.dump(self._add_vec, open(model_file_path, 'wb'))

    def load_model(self, file_prefix: str):
        """
        Load a trained model from files.

        Given the prefix of the file paths, load the model from files with name given by the prefix
        followed by "_embedvecdict.pickle".

        If this has not been run, or a model was not trained by :func:`~train`,
        a `ModelNotTrainedException` will be raised while performing prediction and saving the model.

        :param file_prefix: prefix of the file path
        """
        model_file_path = f"{file_prefix}_embedvecdict.pkl"
        self._add_vec = pickle.load(open(model_file_path, 'rb'))
        self.trained = True

    def score(self, short_text: str) -> Dict[str, float]:
        """
        Calculate the scores for all the class labels for the given short sentence.

        Given a short sentence, calculate the classification scores for all class labels,
        returned as a dictionary with key being the class labels, and values being the scores.
        If the short sentence is empty, or if other numerical errors occur, the score will be `numpy.nan`.

        If neither :func:`~train` nor :func:`~load_model` was run, it will raise `ModelNotTrainedException`.

        :param short_text: a short sentence
        :return: a dictionary with keys being the class labels, and values being the corresponding classification scores
        """
        if not self.trained:
            raise ModelNotTrainedException()
        vec = self._short_text_to_embed_vec(short_text)
        scoredict = {}
        for class_type in self._add_vec:
            try:
                scoredict[class_type] = self.sim_fcn(vec, self._add_vec[class_type])
            except ValueError:
                scoredict[class_type] = np.nan
        return scoredict

    def _short_text_to_embed_vec(self, short_text: str) -> np.ndarray:
        """
        Convert the short text into an averaged embedded vector representation.

        Given a short sentence, it converts all the tokens into embedded vectors according to
        the given word-embedding model, sums them up, and normalize the resulting vector. It returns the resulting vector
        that represents this short sentence.

        :param short_text: a short sentence
        :return: an embedded vector that represents the short sentence
        """
        return shorttext_to_avgvec(short_text, self.wv_model)


def load_sum_word2vec_classifier(wv_model: KeyedVectors, name: str, compact: bool = True, vec_size: int = None) -> SumEmbeddedVecClassifier:
    """
    Load a :class:`shorttext.classifiers.SumEmbeddedVecClassifier` instance from file, given the pre-trained Word2Vec model.

    :param wv_model: Word2Vec model
    :param name: name (if compact=True) or prefix (if compact=False) of the file path
    :param compact whether model file is compact (Default: True)
    :param vec_size: length of embedded vectors in the model (Default: None, directly extracted from word-embedding model)
    :return: the classifier
    """
    classifier = SumEmbeddedVecClassifier(wv_model, vec_size=vec_size)
    if compact:
        classifier.load_compact_model(name)
    else:
        classifier.load_model(name)
    return classifier