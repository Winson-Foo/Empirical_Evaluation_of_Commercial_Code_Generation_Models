from typing import List, Tuple
import time
import numpy as np

from nboost.plugins import Plugin
from nboost.delegates import RequestDelegate, ResponseDelegate
from nboost.helpers import calculate_mrr
from nboost.database import DatabaseRow
from nboost import defaults


class RerankModelPlugin(Plugin):
    """
    Base class for reranker models
    """
    DEFAULT_TOP_K = 10

    def on_request(self, request: RequestDelegate, db_row: DatabaseRow) -> None:
        """
        Sets the top k and top n values in the DatabaseRow and RequestDelegate objects respectively
        """
        db_row.topk = request.topk if request.topk else self.DEFAULT_TOP_K
        request.topk = request.topn

    def on_response(self, response: ResponseDelegate, db_row: DatabaseRow) -> None:
        """
        Reranks the choices using the rank method and updates the response and database row objects
        """
        self._update_server_mrr(response, db_row)
        self._rerank_choices(response)
        self._update_scores(response)
        self._update_model_mrr(response, db_row)
        self._limit_choices(response, db_row.topk)

    def _update_server_mrr(self, response: ResponseDelegate, db_row: DatabaseRow) -> None:
        """
        Calculates and sets the server MRR in the database row object if the request contains rerank_cids
        """
        if response.request.rerank_cids:
            db_row.server_mrr = calculate_mrr(
                correct=response.request.rerank_cids.list,
                guesses=response.cids
            )

    def _rerank_choices(self, response: ResponseDelegate) -> None:
        """
        Reranks the choices using the rank method and sets the choices attribute in the response object
        """
        ranks, scores = self.rank(
            query=response.request.query,
            choices=response.cvalues,
            filter_results=response.request.filter_results
        )

        # remove ranks which are higher than total choices
        ranks = [rank for rank in ranks if rank < len(response.choices)]
        reranked_choices = [response.choices[rank] for rank in ranks]

        response.choices = reranked_choices

    def _update_scores(self, response: ResponseDelegate) -> None:
        """
        Sets the nboost scores in the response object
        """
        response.set_path('body.nboost.scores', list([float(score) for score in scores]))

    def _update_model_mrr(self, response: ResponseDelegate, db_row: DatabaseRow) -> None:
        """
        Calculates and sets the model MRR in the database row object if the request contains rerank_cids
        """
        if response.request.rerank_cids:
            db_row.model_mrr = calculate_mrr(
                correct=response.request.rerank_cids.list,
                guesses=response.cids
            )

    def _limit_choices(self, response: ResponseDelegate, top_k: int) -> None:
        """
        Limits the number of choices in the response object to top_k
        """
        response.choices = response.choices[:top_k]

    @staticmethod
    def rank(query: str, choices: List[str], filter_results: bool = defaults.filter_results) -> Tuple[List[int], List[float]]:
        """
        Assigns relative ranks to each choice
        """
        if len(choices) == 0:
            return [], []

        logits = self.get_logits(query, choices)
        scores = []
        all_scores = []
        index_map = []

        for i, logit in enumerate(logits):
            neg_logit = logit[0]
            score = logit[1]
            all_scores.append(score)
            if score > neg_logit or not filter_results:
                scores.append(score)
                index_map.append(i)

        sorted_indices = [index_map[i] for i in np.argsort(scores)[::-1]]
        return sorted_indices, [all_scores[i] for i in sorted_indices]

    def get_logits(self, query: str, choices: List[str]) -> Tuple:
        """
        Returns the search ranking logits for query, choices.
        """
        raise NotImplementedError()

    def close(self) -> None:
        """
        Closes the model.
        """
        pass
