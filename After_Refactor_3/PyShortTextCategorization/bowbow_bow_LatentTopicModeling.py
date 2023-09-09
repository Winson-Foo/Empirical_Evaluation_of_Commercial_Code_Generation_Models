from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Optional
import numpy as np
from shorttext.utils import textpreprocessing as textpreprocess, gensim_corpora as gc, classification_exceptions as e
from shorttext.utils.textpreprocessing import tokenize

# abstract class to model the latent topics of classdict
class LatentTopicModeler(ABC):
    """
    Abstract class for various topic modeler.
    """
    def __init__(
        self,
        preprocessor: Optional[object] = textpreprocess.standard_text_preprocessor_1(),
        normalize: Optional[bool] = True)->None:
        """ 
        Initialize the modeler.
        :param preprocessor: function that preprocesses the text. (Default: `shorttext.utils.textpreprocess.standard_text_preprocessor_1`)
        :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
        """
        self.preprocessor = preprocessor
        self.normalize = normalize
        self.trained = False

    @abstractmethod
    def train(
        self, 
        classdict: Dict[str, Union[str, Tuple[str, str]]], 
        nb_topics: int, 
        *args, 
        **kwargs)-> None:
        """ 
        Train the modeler.

        :param classdict: training data
        :param nb_topics: number of latent topics
        :param args: arguments to be passed into the wrapped training functions
        :param kwargs: arguments to be passed into the wrapped training functions
        """
        self.nb_topics = nb_topics

    @abstractmethod
    def retrieve_topicvec(
        self, 
        shorttext: str)-> Optional[np.ndarray]:
        """ 
        Calculate the topic vector representation of the short text.
        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param shorttext: short text
        :return: topic vector
        """
        raise e.NotImplementedException()

    @abstractmethod
    def get_batch_cos_similarities(
        self, 
        shorttext: str)-> Optional[np.ndarray]:
        """ 
        Calculate the cosine similarities of the given short text and all the class labels.
        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param shorttext: short text
        :return: topic vector
        """
        raise e.NotImplementedException()

    def generate_gensim_corpora(
        self, 
        classdict: Dict[str, Union[str, Tuple[str, str]]])-> None:
        """ 
        Calculate the gensim dictionary and corpus, and extract the class labels
        from the training data. Called by :func:`~train`.

        :param classdict: training data
        :return: None
        """
        self.dictionary, self.corpus, self.classlabels = gc.generate_gensim_corpora(classdict,
                                                                                    preprocess_and_tokenize=lambda sent: tokenize(self.preprocessor(sent)))

    def get_bow(
        self,
        shorttext: str)-> Optional[Tuple[List[Tuple[int, int]], Dict]]:
        """ 
        Calculate the gensim bag-of-words representation of the given short text string.
        :param shorttext: text to be represented
        :return: corpus representation of the text
        """
        bow = self.dictionary.doc2bow(tokenize(self.preprocessor(shorttext)))
        return bow, self.dictionary

    def get_bow_vector(
        self, 
        shorttext: str, 
        normalize: Optional[bool]=True)-> Optional[np.ndarray]:
        """ 
        Calculate the vector representation of the bag-of-words in terms of numpy.ndarray.
        :param shorttext: short text
        :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
        :return: vector represtation of the text
        """
        if not hasattr(self, 'dictionary') or not hasattr(self, 'preprocessor'):
            raise e.ModelNotTrainedException("Model hasn't been trained yet")
        bow, _ = self.get_bow(shorttext)
        vec = np.zeros(len(self.dictionary))
        for id, val in bow:
            vec[id] = val
        if normalize:
            vec /= np.linalg.norm(vec)
        return vec

    def __getitem__(
        self, 
        shorttext: str)-> Optional[np.ndarray]:
        return self.retrieve_topicvec(shorttext)

    def __contains__(
        self, 
        shorttext: str)-> Optional[bool]:
        """
        Check whether the shorttext is in the corpus.
        """
        if not self.trained:
            raise e.ModelNotTrainedException("Model hasn't been trained yet")
        return True

    @abstractmethod
    def load_model(
        self, 
        nameprefix: str)-> None:
        """ 
        Load the model from files.
        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param nameprefix: prefix of the paths of the model files
        :return: None
        """
        raise e.NotImplementedException()

    @abstractmethod
    def save_model(
        self, 
        nameprefix: str)-> None:
        """ 
        Save the model to files.
        This is an abstract method of this abstract class, which raise the `NotImplementedException`.

        :param nameprefix: prefix of the paths of the model files
        :return: None
        """
        raise e.NotImplementedException()