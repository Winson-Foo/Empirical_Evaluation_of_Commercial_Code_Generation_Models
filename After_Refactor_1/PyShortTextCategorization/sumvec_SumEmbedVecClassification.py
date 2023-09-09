import pickle
from collections import defaultdict
from typing import Any, Dict, List
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

    def __init__(self, wvmodel: KeyedVectors, vecsize: int = None,
                 simfcn: callable = lambda u, v: 1 - cosine(u, v)) -> None:
        """Initialize the classifier.

        Parameters
        ----------
        wvmodel : gensim.models.keyedvectors.KeyedVectors
            Word2Vec model
        vecsize : int, optional
            Length of the embedded vectors in the model (Default: None,
            directly extracted from word-embedding model)
        simfcn : callable, optional
            Similarity function (Default: cosine similarity)
        """
        CompactIOMachine.__init__(self, {'classifier': 'sumvec'}, 'sumvec', ['_embedvecdict.pkl'])
        self.wvmodel = wvmodel
        self.vecsize = self.wvmodel.vector_size if vecsize is None else vecsize
        self.simfcn = simfcn
        self.trained = False

    def train(self, classdict: Dict[str, List[str]]) -> None:
        """Train the classifier.

        If this has not been run, or a model was not loaded by `loadmodel`,
        a `ModelNotTrainedException` will be raised while performing prediction or saving
        the model.

        Parameters
        ----------
        classdict : dict
            Training data
        """
        self.addvec = defaultdict(lambda : np.zeros(self.vecsize))
        for classtype in classdict:
            self.addvec[classtype] = np.sum([self.shorttext_to_embedvec(shorttext)
                                             for shorttext in classdict[classtype]],
                                            axis=0)
            self.addvec[classtype] /= np.linalg.norm(self.addvec[classtype])
        self.addvec = dict(self.addvec)
        self.trained = True

    def savemodel(self, nameprefix: str) -> None:
        """Save the trained model into files.

        Given the prefix of the file paths, save the model into files, with name given by the prefix,
        and add "_embedvecdict.pickle" at the end. If there is no trained model, a `ModelNotTrainedException`
        will be thrown.

        Parameters
        ----------
        nameprefix : str
            Prefix of the file path

        Raises
        ------
        ModelNotTrainedException
            If the model was not trained
        """
        if not self.trained:
            raise ModelNotTrainedException()
        pickle.dump(self.addvec, open(nameprefix + '_embedvecdict.pkl', 'wb'))

    def loadmodel(self, nameprefix: str) -> None:
        """Load a trained model from files.

        Given the prefix of the file paths, load the model from files with name given by the prefix
        followed by "_embedvecdict.pickle".

        If this has not been run, or a model was not trained by `train`,
        a `ModelNotTrainedException` will be raised while performing prediction and saving the model.

        Parameters
        ----------
        nameprefix : str
            Prefix of the file path

        Raises
        ------
        ModelNotTrainedException
            If the model was not trained
        """
        self.addvec = pickle.load(open(nameprefix + '_embedvecdict.pkl', 'rb'))
        self.trained = True

    def shorttext_to_embedvec(self, shorttext: str) -> np.ndarray:
        """Convert the short text into an averaged embedded vector representation.

        Given a short sentence, it converts all the tokens into embedded vectors according to
        the given word-embedding model, sums them up, and normalize the resulting vector.
        It returns the resulting vector that represents this short sentence.

        Parameters
        ----------
        shorttext : str
            A short sentence

        Returns
        -------
        numpy.ndarray
            An embedded vector that represents the short sentence
        """
        return shorttext_to_avgvec(shorttext, self.wvmodel)

    def score(self, shorttext: str) -> Dict[str, float]:
        """Calculate the scores for all the class labels for the given short sentence.

        Given a short sentence, calculate the classification scores for all class labels,
        returned as a dictionary with key being the class labels, and values being the scores.
        If the short sentence is empty, or if other numerical errors occur, the score will be `numpy.nan`.

        If neither `train` nor `loadmodel` was run, it will raise `ModelNotTrainedException`.

        Parameters
        ----------
        shorttext : str
            A short sentence

        Returns
        -------
        dict
            A dictionary with keys being the class labels, and values being the corresponding classification scores

        Raises
        ------
        ModelNotTrainedException
            If the model was not trained
        """
        if not self.trained:
            raise ModelNotTrainedException()
        vec = self.shorttext_to_embedvec(shorttext)
        scoredict = {}
        for classtype in self.addvec:
            try:
                scoredict[classtype] = self.simfcn(vec, self.addvec[classtype])
            except ValueError:
                scoredict[classtype] = np.nan
        return scoredict


def load_sumword2vec_classifier(wvmodel: KeyedVectors, name: str, compact: bool = True,
                                vecsize: int = None) -> SumEmbeddedVecClassifier:
    """Load a `SumEmbeddedVecClassifier` instance from file, given the pre-trained Word2Vec model.

    Parameters
    ----------
    wvmodel : gensim.models.keyedvectors.KeyedVectors
        Word2Vec model
    name : str
        Name (if compact=True) or prefix (if compact=False) of the file path
    compact : bool, optional
        Whether model file is compact (Default: True)
    vecsize : int, optional
        Length of embedded vectors in the model (Default: None,
        directly extracted from word-embedding model)

    Returns
    -------
    SumEmbeddedVecClassifier
        The classifier
    """
    classifier = SumEmbeddedVecClassifier(wvmodel, vecsize=vecsize)
    if compact:
        classifier.load_compact_model(name)
    else:
        classifier.loadmodel(name)
    return classifier