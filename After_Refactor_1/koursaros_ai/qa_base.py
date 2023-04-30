from typing import Tuple
from nboost.plugins import Plugin
from nboost.delegates import ResponseDelegate
from nboost import defaults


class QAModelPlugin(Plugin):
    def __init__(
        self,
        max_query_length: int = defaults.max_query_length,
        model_dir: str = defaults.qa_model_dir,
        max_seq_len: int = defaults.max_seq_len,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dir = model_dir
        self.max_query_length = max_query_length
        self.max_seq_len = max_seq_len

    def on_response(self, response: ResponseDelegate, db_row) -> None:
        if response.cvalues:
            answer, start_pos, end_pos, score, time_taken = self.get_answer(
                query=response.request.query,
                context=response.cvalues[0],
                qa_threshold=response.request.qa_threshold,
            )

            if score > defaults.qa_score_threshold:
                response.set_path("body.nboost.answer_text", answer)
                response.set_path("body.nboost.answer_start_pos", start_pos)
                response.set_path("body.nboost.answer_stop_pos", end_pos)

            logging.info(
                f"Answer: '{answer}' (score: {score:.2f}, time: {time_taken:.2f}s)"
            )

    def get_answer(
        self, query: str, context: str, qa_threshold: float
    ) -> Tuple[str, int, int, float, float]:
        """Return answer, start_pos, end_pos, score, time_taken"""
        start_time = time.perf_counter()

        # TODO: Implement QA model logic here.

        time_taken = time.perf_counter() - start_time
        return "sample answer", 0, 10, 0.8, time_taken  # TODO: Replace with real values