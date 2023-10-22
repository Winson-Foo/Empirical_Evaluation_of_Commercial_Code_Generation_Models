import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from typing import List, Tuple
from nboost.plugins.models.rerank.base import RerankModelPlugin
from nboost import defaults

class USERerankModelPlugin(RerankModelPlugin):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.module = hub.load(self.model_dir)

    def get_question_embedding(self, query: str) -> np.ndarray:
        question_embedding = self.module([query])
        return question_embedding

    def get_candidate_embeddings(self, candidate_responses: List[str]) -> np.ndarray:
        candidate_embeddings = self.module(candidate_responses)
        return candidate_embeddings

    def calculate_scores(self, question_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        scores = np.inner(question_embedding, candidate_embeddings)
        scores = np.reshape(scores, (-1,))
        return scores

    def sort_results(self, scores: np.ndarray) -> Tuple[List[int], List[float]]:
        sorted_indices = list(np.argsort(scores)[::-1])
        return sorted_indices, scores[sorted_indices]

    def rank(self, query: str, candidate_responses: List[str],
             filter_results: type(defaults.filter_results) = defaults.filter_results
             ) -> Tuple[List[int], List[float]]:

        question_embedding = self.get_question_embedding(query)
        candidate_embeddings = self.get_candidate_embeddings(candidate_responses)
        scores = self.calculate_scores(question_embedding, candidate_embeddings)
        sorted_indices, sorted_scores = self.sort_results(scores)
        return sorted_indices, sorted_scores
