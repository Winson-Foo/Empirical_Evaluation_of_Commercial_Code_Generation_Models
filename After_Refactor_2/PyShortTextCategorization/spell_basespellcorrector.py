class BaseSpellCorrector(ABC):
    """Abstract base class for all spell correctors."""

    @abstractmethod
    def train(self, corpus: str) -> None:
        """Train the spell corrector with the given corpus.

        :param corpus: training corpus
        :type corpus: str
        """
        pass

    @abstractmethod
    def correct(self, word: str) -> str:
        """Recommend a spell correction for the given word.

        :param word: word to be corrected
        :type word: str
        :return: recommended correction
        :rtype: str
        """
        pass