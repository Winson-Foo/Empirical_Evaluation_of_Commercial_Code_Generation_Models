from abc import ABC, abstractmethod
import numpy as np
from shorttext.utils import textpreprocessing as textpreprocess, gensim_corpora as gc, classification_exceptions as e
from shorttext.utils.textpreprocessing import tokenize


class CorpusGenerator:
    """
    Given training data, generate gensim dictionary and corpus.
    """
    def __init__(self, preprocessor=textpreprocess.standard_text_preprocessor_1()):
        self.preprocessor = preprocessor
        
    def generate(self, classdict):
        """
        :param classdict: dict, training data
        :return dictionary: gensim.corpora.Dictionary
        :return corpus: list of list of tuples, a list of the corpus bag-of-words representation where each tuple is (word_id, count)
        :return classlabels: list, extracted class labels from the training data
        """
        dictionary, corpus, classlabels = gc.generate_gensim_corpora(
            classdict,
            preprocess_and_tokenize=lambda sent: tokenize(self.preprocessor(sent)))
        return dictionary, corpus, classlabels


class LatentTopicModeler(ABC):
    """
    Abstract class for various topic modeler.
    """
    def __init__(self, normalize=True):
        """
        :param normalize: bool, whether to normalize the retrieved topic vectors (default: True)
        """
        self.normalize = normalize
        self.trained = False

    def train(self, classdict, nb_topics):
        """
        Train the modeler.
        :param classdict: dict, training data
        :param nb_topics: int, number of latent topics
        """
        self.dictionary, self.corpus, self.classlabels = CorpusGenerator().generate(classdict)
        self.nb_topics = nb_topics
        self._train()
        self.trained = True

    @abstractmethod
    def _train(self):
        """
        Train the model
        """
        pass

    def retrieve_bow_vector(self, shorttext):
        """
        Calculate the vector representation of the bag-of-words in terms of numpy.ndarray.
        :param shorttext: str, short text
        :return: ndarray, vector represtation of the text
        """
        bow = self.retrieve_bow(shorttext)
        vec = np.zeros(len(self.dictionary))
        for id, val in bow:
            vec[id] = val
        if self.normalize:
            vec /= np.linalg.norm(vec)
        return vec

    def retrieve_topicvec(self, shorttext):
        """
        Calculate the topic vector representation of the short text.
        :param shorttext: str, short text
        :return: ndarray, topic vector
        """
        vec = self.retrieve_bow_vector(shorttext)
        topicvec = self._retrieve_topicvec(vec)
        if self.normalize:
            topicvec /= np.linalg.norm(topicvec)
        return topicvec

    @abstractmethod
    def _retrieve_topicvec(self, vec):
        """
        Calculate the topic vector representation of the given vector
        :param vec: ndarray, vector representation of a short text
        :return: ndarray, topic vector representation
        """
        pass

    @abstractmethod
    def get_batch_cos_similarities(self, shorttext):
        """
        Calculate the cosine similarities of the given short text and all the class labels.
        :param shorttext: str, short text
        :return: ndarray, cosine similarities
        """
        pass

    def __getitem__(self, shorttext):
        """
        Get the topic vector representation of a short text
        :param shorttext: str, short text
        :return: ndarray, topic vector representation
        """
        return self.retrieve_topicvec(shorttext)

    def __contains__(self, shorttext):
        """
        Check if the model has been trained
        :param shorttext: str, ignored
        :return: bool, whether the model has been trained
        """
        return self.trained

    @abstractmethod
    def loadmodel(self, nameprefix):
        """
        Load the model from files.
        :param nameprefix: str, prefix of the paths of the model files
        """
        pass

    @abstractmethod
    def savemodel(self, nameprefix):
        """
        Save the model to files.
        :param nameprefix: str, prefix of the paths of the model files
        """
        pass