import re
from collections import Counter
from typing import List, Set

from . import SpellCorrector
from .editor import compute_set_edits1, compute_set_edits2


class NorvigSpellCorrector(SpellCorrector):
    """ Spell corrector described by Peter Norvig in his blog. (https://norvig.com/spell-correct.html)
    """
    def __init__(self):
        """ Instantiate the class and train the spell corrector.
        """
        self.train('')

    def train(self, text: str) -> None:
        """ Given the training corpus, train the spell corrector.

        :param text: training corpus
        :type text: str
        """
        self.words = re.findall('\\w+', text.lower())
        self.WORDS = Counter(self.words)
        self.N = sum(self.WORDS.values())

    def P(self, word: str) -> float:
        """ Compute the probability of the word sampled randomly from the training corpus.

        :param word: a word
        :return: probability of the word sampled randomly in the corpus
        :rtype: float
        """
        return self.WORDS[word] / float(self.N)

    def correct(self, word: str) -> str:
        """ Recommend a spelling correction to the given word.

        :param word: a word
        :return: recommended correction
        :rtype: str
        """
        return max(self.candidates(word), key=self.P)

    def known(self, words: List[str]) -> Set[str]:
        """ Filter the words that can be found in the training corpus.

        :param words: list of words
        :return: set of words that can be found in the training corpus
        :rtype: set
        """
        return set(w for w in words if w in self.WORDS)

    def candidates(self, word: str) -> List[str]:
        """ List potential candidates for corrected spelling to the given word.

        :param word: a word
        :return: list of recommended corrections
        :rtype: list
        """
        return (
            self.known([word])
            or self.known(compute_set_edits1(word))
            or self.known(compute_set_edits2(word))
            or [word]
        )