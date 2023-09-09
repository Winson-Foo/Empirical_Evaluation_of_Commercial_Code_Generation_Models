import re
from collections import Counter
from typing import List, Set


class NorvigSpellCorrector:
    """
    Peter Norvig's Spell Corrector implementation.

    Reference: https://norvig.com/spell-correct.html
    """

    def __init__(self) -> None:
        """
        Initialize the spell corrector.
        """
        self.words: List[str] = []
        self.WORDS: Counter = Counter()
        self.N: int = 0

    def train(self, text: str) -> None:
        """
        Train the spell corrector with the given text corpus.

        :param text: Text corpus to train on.
        """
        self.words = re.findall('\\w+', text.lower())
        self.WORDS = Counter(self.words)
        self.N = sum(self.WORDS.values())

    def probability(self, word: str) -> float:
        """
        Compute the probability of a word based on its frequency in the training corpus.

        :param word: The word to compute the probability for.
        :return: The probability of the word in the training corpus.
        """
        return self.WORDS[word] / float(self.N)

    def correct(self, word: str) -> str:
        """
        Correct a misspelled word by suggesting a correction candidate.

        :param word: The misspelled word to correct.
        :return: The suggested correction candidate.
        """
        return max(self.candidates(word), key=self.probability)

    def known_words(self, words: List[str]) -> Set[str]:
        """
        Filter the list of words to only include those that are known in the training corpus.

        :param words: The list of words to filter.
        :return: A set of words that are known in the training corpus.
        """
        return set(word for word in words if word in self.WORDS)

    def candidates(self, word: str) -> List[str]:
        """
        Generate a list of correction candidates for a misspelled word.

        :param word: The misspelled word to generate candidates for.
        :return: A list of correction candidates.
        """
        candidates = (
            self.known_words([word])
            or self.known_words(compute_set_edits1(word))
            or self.known_words(compute_set_edits2(word))
            or [word]
        )
        return list(candidates)


def compute_set_edits1(word: str) -> Set[str]:
    """
    Computes the set of all possible corrections that are one edit away from the given word.

    :param word: The word to generate corrections for.
    :return: A set of potential correction candidates.
    """
    # implementation details


def compute_set_edits2(word: str) -> Set[str]:
    """
    Computes the set of all possible corrections that are two edits away from the given word.

    :param word: The word to generate corrections for.
    :return: A set of potential correction candidates.
    """
    # implementation details