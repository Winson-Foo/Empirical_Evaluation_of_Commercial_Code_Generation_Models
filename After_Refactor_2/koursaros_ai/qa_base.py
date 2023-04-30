from typing import Tuple
import time
from nboost.plugins import Plugin
from nboost.delegates import ResponseDelegate
from nboost.database import DatabaseRow
from nboost import defaults
import logging

logger = logging.getLogger(__name__)

class QAModelPlugin(Plugin):
    def __init__(self,
                 max_query_length: type(defaults.max_query_length) = defaults.max_query_length,
                 model_dir: str = defaults.qa_model_dir,
                 max_seq_len: int = defaults.max_seq_len,
                 **kwargs):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self.max_query_length = max_query_length
        self.max_seq_len = max_seq_len

    def on_response(self, response: ResponseDelegate, db_row: DatabaseRow):
        if response.cvalues:
            start_time = time.perf_counter()

            answer, start_pos, stop_pos, score = self.get_answer(response.request.query, response.cvalues[0])

            db_row.qa_time = time.perf_counter() - start_time

            logger.info(f"Answer: {answer}, Start pos: {start_pos}, Stop pos: {stop_pos}, Score: {score}")
            
            if score > response.request.qa_threshold:
                self.set_response_body(response, answer, start_pos, stop_pos)

    def get_answer(self, query: str, cvalue: str) -> Tuple[str, int, int, float]:
        """Return answer, start_pos, end_pos, score"""
        raise NotImplementedError()

    def set_response_body(self, response: ResponseDelegate, answer: str, start_pos: int, stop_pos: int):
        response.set_path('body.nboost.answer_text', answer)
        response.set_path('body.nboost.answer_start_pos', start_pos)
        response.set_path('body.nboost.answer_stop_pos', stop_pos) 