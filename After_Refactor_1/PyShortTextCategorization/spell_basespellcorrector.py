from abc import ABC, abstractmethod

class SpellCorrector(ABC):
    """ Base class for all spell corrector. """

    @abstractmethod
    def train(self, text):
        """ Train the spell corrector with the given corpus.

        :param text: training corpus
        :type text: str
        """
        pass

    @abstractmethod
    def correct(self, word):
        """ Recommend a spell correction to given the word.

        :param word: word to be checked
        :return: recommended correction
        :type word: str
        :rtype: str
        """
        pass

class NotImplementedException(Exception):
    """ Exception class for not implemented methods """
    pass