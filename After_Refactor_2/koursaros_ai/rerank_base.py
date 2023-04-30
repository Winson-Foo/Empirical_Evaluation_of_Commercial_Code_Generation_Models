from typing import List, Tuple
import time
import numpy as np
from nboost.plugins import Plugin
from nboost.delegates import RequestDelegate, ResponseDelegate
from nboost.helpers import calculate_mrr
from nboost.database import DatabaseRow
from nboost import defaults

class RerankModelPlugin(Plugin):
    """Base class for reranker models"""

    def on_request(self, request: RequestDelegate, db_row: DatabaseRow) -> None:
        """Modify the request object before sending it to the backend"""
        db_row.topk = request.topk if request.topk else request.default_topk
        request.topk = request.topn

    def on_response(self, response: ResponseDelegate, db_row: DatabaseRow) -> None:
        """Modify the response object returned from the backend"""
        if response.request.rerank_cids:
            db_row.server_mrr = calculate_mrr(
                correct=response.request.rerank_cids.list,
                guesses=response.cids
            )

        start_time = time.perf_counter()

        ranks, scores = self.rank(
            query=response.request.query,
            choices=response.cvalues,
            filter_results=response.request.filter_results
        )
        db_row.rerank_time = time.perf_counter() - start_time

        # remove ranks which are higher than total choices
        ranks = [rank for rank in ranks if rank < len(response.choices)]
        reranked_choices = [response.choices[rank] for rank in ranks]

        response.choices = reranked_choices
        response.set_path('body.nboost.scores', list(map(float, scores)))

        if response.request.rerank_cids:
            db_row.model_mrr = calculate_mrr(
                correct=response.request.rerank_cids.list,
                guesses=response.cids
            )

        response.choices = response.choices[:db_row.topk]

    def rank(self, query: str, choices: List[str], filter_results: bool = defaults.filter_results) -> Tuple[List[int], List[float]]:
        """Assign relative ranks to each choice"""
        if not choices:
            return [], []

        logits = self.get_logits(query, choices)
        scores = []
        all_scores = []
        index_map = []
        for i, (neg_logit, score) in enumerate(logits):
            all_scores.append(score)
            if score > neg_logit or not filter_results:
                scores.append(score)
                index_map.append(i)

        sorted_indices = [index_map[i] for i in np.argsort(scores)[::-1]]
        return sorted_indices, [all_scores[i] for i in sorted_indices]

    def get_logits(self, query: str, choices: List[str]) -> List[Tuple[float, float]]:
        """Get search ranking logits for query, choices"""
        raise NotImplementedError()

    def close(self) -> None:
        """Close the model"""
        pass
