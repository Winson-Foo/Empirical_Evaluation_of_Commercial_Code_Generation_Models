import pickle
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cosine

from shorttext.utils.classification_exceptions import ModelNotTrainedException
from shorttext.utils import shorttext_to_avgvec
from shorttext.utils.compactmodel_io import CompactIOMachine

WORD2VEC_MODEL_URL = 'https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit'
DEFAULT_SIMILARITY_FUNCTION = lambda u, v: 1 - cosine(u, v)
DEFAULT_VEC_SIZE = None


class SumEmbeddedVecClassifier(CompactIOMachine):
    """
    This is a supervised classification algorithm for short text categorization.
    Each class label has a few short sentences, where each token is converted
    to an embedded vector, given by a pre-trained word-embedding model (e.g., Google Word2Vec model).
    They are then summed up and normalized to a unit vector for that particular class labels.
    To perform prediction, the input short sentences is converted to a unit vector
    in the same way. The similarity score is calculated by the cosine similarity.

    A pre-trained Google Word2Vec model can be downloaded from this URL: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit.
    """

    def __init__(self, wvmodel, vec_size=DEFAULT_VEC_SIZE, sim_fcn=DEFAULT_SIMILARITY_FUNCTION):
        """ Initialize the classifier.

        :param wvmodel: Word2Vec model
        :param vec_size: length of the embedded vectors in the model (Default: None, directly extracted from word-embedding model)
        :param sim_fcn: similarity function (Default: cosine similarity)
        :type wvmodel: gensim.models.keyedvectors.KeyedVectors
        :type vec_size: int
        :type sim_fcn: function
        """
        CompactIOMachine.__init__(self, {'classifier': 'sumvec'}, 'sumvec', ['_embedvecdict.pkl'])
        self.word2vec_model = wvmodel
        self.vec_size = self.word2vec_model.vector_size if vec_size is None else vec_size
        self.sim_fcn = sim_fcn
        self.is_trained = False

    def train(self, class_dict):
        """ Train the classifier.

        If this has not been run, or a model was not loaded by :func:`~load_model`,
        a `ModelNotTrainedException` will be raised while performing prediction or saving
        the model.

        :param class_dict: training data
        :type class_dict: dict
        """
        self.add_vec = defaultdict(lambda: np.zeros(self.vec_size))
        for class_type in class_dict:
            self.add_vec[class_type] = np.sum([self.short_text_to_embedded_vec(short_text)
                                               for short_text in class_dict[class_type]], axis=0)
            self.add_vec[class_type] /= np.linalg.norm(self.add_vec[class_type])
        self.add_vec = dict(self.add_vec)
        self.is_trained = True

    def save_model(self, file_prefix):
        """ Save the trained model into files.

        Given the prefix of the file paths, save the model into files, with name given by the prefix,
        and add "_embedvecdict.pickle" at the end. If there is no trained model, a `ModelNotTrainedException`
        will be thrown.

        :param file_prefix: prefix of the file path
        :type file_prefix: str
        :raise: ModelNotTrainedException
        """
        if not self.is_trained:
            raise ModelNotTrainedException()
        with open(f"{file_prefix}_embedvecdict.pkl", "wb") as f:
            pickle.dump(self.add_vec, f)

    def load_model(self, file_prefix):
        """ Load a trained model from files.

        Given the prefix of the file paths, load the model from files with name given by the prefix
        followed by "_embedvecdict.pickle".

        If this has not been run, or a model was not trained by :func:`~train`,
        a `ModelNotTrainedException` will be raised while performing prediction and saving the model.

        :param file_prefix: prefix of the file path
        :type file_prefix: str
        """
        with open(f"{file_prefix}_embedvecdict.pkl", "rb") as f:
            self.add_vec = pickle.load(f)
            self.is_trained = True

    def short_text_to_embedded_vec(self, short_text):
        """ Convert the short text into an averaged embedded vector representation.

        Given a short sentence, it converts all the tokens into embedded vectors according to
        the given word-embedding model, sums
        them up, and normalize the resulting vector. It returns the resulting vector
        that represents this short sentence.

        :param short_text: a short sentence
        :type short_text: str
        :rtype: numpy.ndarray
        """
        return shorttext_to_avgvec(short_text, self.word2vec_model)

    def score(self, short_text):
        """ Calculate the scores for all the class labels for the given short sentence.

        Given a short sentence, calculate the classification scores for all class labels,
        returned as a dictionary with key being the class labels, and values being the scores.
        If the short sentence is empty, or if other numerical errors occur, the score will be `numpy.nan`.

        If neither :func:`~train` nor :func:`~load_model` was run, it will raise `ModelNotTrainedException`.

        :param short_text: a short sentence
        :type short_text: str
        :rtype: dict
        :raise: ModelNotTrainedException
        """
        if not self.is_trained:
            raise ModelNotTrainedException()
        vec = self.short_text_to_embedded_vec(short_text)
        score_dict = {}
        for class_type in self.add_vec:
            try:
                score_dict[class_type] = self.sim_fcn(vec, self.add_vec[class_type])
            except ValueError:
                score_dict[class_type] = np.nan
        return score_dict


def load_sum_word2vec_classifier(wvmodel, name_prefix, compact=True, vec_size=DEFAULT_VEC_SIZE):
    """ Load a :class:`shorttext.classifiers.SumEmbeddedVecClassifier` instance from file, given the pre-trained Word2Vec model.

    :param wvmodel: Word2Vec model
    :param name_prefix: name (if compact=True) or prefix (if compact=False) of the file path
    :param compact: whether model file is compact (Default: True)
    :param vec_size: length of embedded vectors in the model (Default: None, directly extracted from word-embedding model)
    :return: the classifier
    :rtype: SumEmbeddedVecClassifier
    """
    classifier = SumEmbeddedVecClassifier(wvmodel, vec_size=vec_size)
    if compact:
        classifier.load_compact_model(name_prefix)
    else:
        classifier.load_model(name_prefix)
    return classifier