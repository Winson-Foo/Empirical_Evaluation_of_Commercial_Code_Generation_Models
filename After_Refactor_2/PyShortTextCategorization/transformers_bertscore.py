from itertools import product
from typing import List

import torch
from torch.nn import CosineSimilarity

from ...utils.transformers import WrappedBERTEncoder


class BERTScorer:
    """Compute the BERT scores between two sentences."""
    MAX_LENGTH = 48
    NB_ENCODING_LAYERS = 4

    def __init__(
        self,
        model: str = None,
        tokenizer: str = None,
        max_length: int = MAX_LENGTH,
        nb_encoding_layers: int = NB_ENCODING_LAYERS,
        device: str = 'cpu'
    ) -> None:
        """Initialize the BERT scorer."""

        self.encoder = WrappedBERTEncoder(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            nbencodinglayers=nb_encoding_layers,
            device=device
        )

        self.device = self.encoder.device
        self.cosine_fcn = CosineSimilarity(dim=0).to(self.device)

    def compute_matrix(
        self,
        reference_sentence: str,
        test_sentence: str
    ) -> torch.Tensor:
        """Compute a similarity matrix between the tokens in two sentences."""

        similarity_matrix = torch.zeros(
            (self.MAX_LENGTH - 2, self.MAX_LENGTH - 2),
            device=self.device
        )

        _, ref_embeddings, _ = self.encoder.encode_sentences(
            [reference_sentence]
        )
        _, test_embeddings, _ = self.encoder.encode_sentences([test_sentence])

        cos = self.cosine_fcn
        for i, j in product(range(len(ref_embeddings[0]) - 2),
                            range(len(test_embeddings[0]) - 2)):
            similarity_matrix[i, j] = cos(
                ref_embeddings[0][i + 1],
                test_embeddings[0][j + 1]
            )

        return similarity_matrix

    def _bertscore(
        self,
        reference_sentence: str,
        test_sentence: str,
        axis: int = 1
    ) -> float:
        """Compute either recall or precision BERT score."""

        similarity_matrix = self.compute_matrix(reference_sentence, test_sentence)
        score = torch.mean(torch.max(similarity_matrix, axis=axis).values)
        return score.item()

    def recall_bertscore(
        self,
        reference_sentence: str,
        test_sentence: str
    ) -> float:
        """Compute the recall BERT score."""

        return self._bertscore(reference_sentence, test_sentence, axis=1)

    def precision_bertscore(
        self,
        reference_sentence: str,
        test_sentence: str
    ) -> float:
        """Compute the precision BERT score."""

        return self._bertscore(reference_sentence, test_sentence, axis=0)

    def f1score_bertscore(
        self,
        reference_sentence: str,
        test_sentence: str
    ) -> float:
        """Compute the F1 BERT score."""

        recall = self.recall_bertscore(reference_sentence, test_sentence)
        precision = self.precision_bertscore(reference_sentence, test_sentence)

        if recall + precision == 0:
            return 0
        else:
            return 2 * recall * precision / (recall + precision)