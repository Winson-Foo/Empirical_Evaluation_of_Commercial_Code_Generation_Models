import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import List, Tuple
from nboost import defaults
from nboost.plugins.models.rerank.base import RerankModelPlugin

class USERerankModelPlugin(RerankModelPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module = hub.load(self.model_dir)

    def rank(self, query: str, choices: List[str], filter_results: type(defaults.filter_results) = defaults.filter_results) -> Tuple[List[int], List[float]]:
        # Embed the query and choices using the pre-trained module
        question_embedding = self.module([query])
        candidate_embeddings = self.module(choices)

        # Compute cosine similarity scores between the query and each choice
        scores = np.inner(question_embedding, candidate_embeddings)
        scores = np.reshape(scores, (-1,))

        # Sort the choices by their similarity score to the query
        sorted_indices = list(np.argsort(scores)[::-1])

        # Return the sorted choice indices and their respective scores
        return sorted_indices, scores[sorted_indices]
