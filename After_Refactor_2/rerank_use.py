import tensorflow_hub as hub
import numpy as np
from typing import List, Tuple

from nboost.plugins.models.rerank.base import RerankModelPlugin
from nboost import defaults


class TensorFlowHubModel:
    """
    Encapsulates the functionality for loading and encoding with a TensorFlow Hub model.
    """

    def __init__(self, model_dir: str):
        self.module = hub.load(model_dir)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a given query into a fixed-size embedding vector.
        """
        return self.module([query])

    def encode_choices(self, choices: List[str]) -> np.ndarray:
        """
        Encodes a list of choices (candidate responses) into a matrix of fixed-size
        embedding vectors.
        """
        return self.module(choices)



class USERerankModelPlugin(RerankModelPlugin):
    """
    A reranking model that uses a TensorFlow Hub model to score candidate responses
    for a given query.
    """

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(**kwargs)
        self.tf_model = TensorFlowHubModel(model_dir)

    def rank(self, query: str, choices: List[str],
             filter_results: type(defaults.filter_results) = defaults.filter_results
             ) -> Tuple[List[int], List[float]]:
        query_embedding = self.tf_model.encode_query(query)
        candidate_embeddings = self.tf_model.encode_choices(choices)

        scores = np.inner(query_embedding, candidate_embeddings)
        sorted_indices = list(np.argsort(scores)[::-1])
        sorted_scores = scores[sorted_indices]

        return sorted_indices, sorted_scores
