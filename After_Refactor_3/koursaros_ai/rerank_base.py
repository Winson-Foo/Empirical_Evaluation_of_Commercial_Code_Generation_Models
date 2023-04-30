from typing import List, Tuple
import time
from nboost.plugins import Plugin
from nboost.database import DatabaseRow
from nboost import defaults
from nboost.helpers import calculate_mrr
from nboost.rerank.common import rank_choices
import numpy as np
from typing import List, Tuple


class RerankModelPlugin(Plugin):
    """Base class for reranker models"""

    def on_request(self, request, db_row):
        """Set topk to default_topk if topk is None and replace topk with topn."""
        db_row.topk = request.topk or request.default_topk
        request.topk = request.topn

    def on_response(self, response, db_row):
        """Rank the choices and rerank them based on the scores."""
        if response.request.rerank_cids:
            db_row.server_mrr = calculate_mrr(
                correct=response.request.rerank_cids.list,
                guesses=response.cids
            )

        start_time = time.perf_counter()

        ranks, scores = self.rank(response.request.query, response.cvalues,
                                  response.request.filter_results)
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

    def rank(self, query: str, choices: List[str], filter_results: bool
             ) -> Tuple[List[int], List[float]]:
        """Assign relative ranks to each choice."""
        if len(choices) == 0:
            return [], []

        logits = self.get_logits(query, choices)
        return rank_choices(logits, filter_results)

    def get_logits(self, query: str, choices: List[str]) -> List[Tuple[float, float]]:
        """Get search ranking logits for query, choices."""
        raise NotImplementedError()

    def close(self):
        """Close the model."""
        pass

def rank_choices(logits: List[Tuple[float, float]], filter_results: bool
                 ) -> Tuple[List[int], List[float]]:
    """Assign relative ranks to each choice."""
    scores, index_map, all_scores = [], [], []
    for i, (neg_logit, score) in enumerate(logits):
        all_scores.append(score)
        if score > neg_logit or not filter_results:
            scores.append(score)
            index_map.append(i)
    sorted_indices = [index_map[i] for i in np.argsort(scores)[::-1]]
    return sorted_indices, [all_scores[i] for i in sorted_indices]