import time
from typing import Tuple

from nboost.plugins import Plugin
from nboost.delegates import ResponseDelegate
from nboost.database import DatabaseRow
from nboost import defaults


class QAModelPlugin(Plugin):
    def __init__(self,
                 max_query_length: int = defaults.max_query_length,
                 model_dir: str = defaults.qa_model_dir,
                 max_seq_len: int = defaults.max_seq_len,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self.max_query_length = max_query_length
        self.max_seq_len = max_seq_len

    def on_response(self, response: ResponseDelegate, db_row: DatabaseRow) -> None:
        if response.cvalues:
            start_time = time.perf_counter()

            answer, start_pos, stop_pos, score = self.get_answer(response.request.query, response.cvalues[0])

            db_row.qa_time = time.perf_counter() - start_time

            self.logger.debug(f"Answer: {answer}, Start Pos: {start_pos}, Stop Pos: {stop_pos}, Score: {score}")
            if score > response.request.qa_threshold:
                response.set_path('body.nboost.answer_text', answer)
                response.set_path('body.nboost.answer_start_pos', start_pos)
                response.set_path('body.nboost.answer_stop_pos', stop_pos)

    def get_answer(self, query: str, cvalue: str) -> Tuple[str, int, int, float]:
        """Return answer, start_pos, end_pos, score."""
        raise NotImplementedError()
