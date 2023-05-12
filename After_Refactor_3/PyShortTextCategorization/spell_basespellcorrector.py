from abc import ABC, abstractmethod


class SpellCorrector(ABC):
    """ Base class for all spell correctors. """

    @abstractmethod
    def train_corpus(self, corpus):
        """ Train the spell corrector with the given corpus. """
        pass

    @abstractmethod
    def correct_word(self, word):
        """ Recommend a spell correction to given the word. """
        return word


class MySpellCorrector(SpellCorrector):
    """ Example implementation of SpellCorrector class. """

    def train_corpus(self, corpus):
        """ Train the spell corrector with the given corpus. """
        # implementation logic here
        pass

    def correct_word(self, word):
        """ Recommend a spell correction to given the word. """
        # implementation logic here
        return word