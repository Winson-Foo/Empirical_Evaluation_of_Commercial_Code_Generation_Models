from itertools import product

import numpy as np
import torch

from ...utils.transformers import WrappedBERTEncoder


class BERTScorer:
    """Compute BERTScores between sentences."""
    def __init__(
            self,
            model: str = None,
            tokenizer: str = None,
            max_length: int = 48,
            nb_encoding_layers: int = 4,
            device: str = 'cpu'
    ):
        """
        :param model: BERT model (default: None, with model `bert-base-uncase` to be used)
        :param tokenizer: BERT tokenizer (default: None, with model `bert-base-uncase` to be used)
        :param max_length: maximum number of tokens of each sentence (default: 48)
        :param nb_encoding_layers: number of encoding layers (taking the last layers to encode the sentences, default: 4)
        :param device: device the language model is stored (default: `cpu`)
        """
        self.encoder = WrappedBERTEncoder(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            nbencodinglayers=nb_encoding_layers,
            device=device)
        self.device = self.encoder.device
        self.cosine_fcn = torch.nn.CosineSimilarity(dim=0).to(self.device)

    def compute_similarity(self, sentence_a: str, sentence_b: str) -> np.ndarray:
        """Compute the table of similarities between all pairs of tokens."""
        cos = self.cosine_fcn
        _, sentence_a_embeddings, sentence_a_tokens = self.encoder.encode_sentences([sentence_a])
        _, sentence_b_embeddings, sentence_b_tokens = self.encoder.encode_sentences([sentence_b])

        similarity_matrix = torch.zeros((len(sentence_a_tokens[0])-2, len(sentence_b_tokens[0])-2),
                                        device=self.device)

        for i, j in product(range(len(sentence_a_tokens[0])-2), range(len(sentence_b_tokens[0])-2)):
            similarity_matrix[i, j] = cos(sentence_a_embeddings[0][i+1],
                                          sentence_b_embeddings[0][j+1])

        return similarity_matrix

    def compute_recall(self, reference: str, test: str) -> float:
        """Compute the recall BERTScore between two sentences."""
        similarity_matrix = self.compute_similarity(reference, test)
        return np.float(torch.mean(torch.max(similarity_matrix, axis=1).values).detach().numpy())

    def compute_precision(self, reference: str, test: str) -> float:
        """Compute the precision BERTScore between two sentences."""
        similarity_matrix = self.compute_similarity(reference, test)
        return np.float(torch.mean(torch.max(similarity_matrix, axis=0).values).detach().numpy())

    def compute_f1_score(self, reference: str, test: str) -> float:
        """Compute the F1 BERTScore between two sentences."""
        recall = self.compute_recall(reference, test)
        precision = self.compute_precision(reference, test)
        return 2 * recall * precision / (recall + precision)