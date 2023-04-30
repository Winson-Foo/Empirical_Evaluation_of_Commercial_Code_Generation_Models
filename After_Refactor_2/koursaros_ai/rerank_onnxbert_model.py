import glob
import onnxruntime as rt
import numpy as np
from typing import List
from config import ModelConfig
from tokenizer import Tokenizer

class Model:
    """
    Wrapper for ONNX Rerank Model from Microsoft.
    """

    def __init__(self, model_config=ModelConfig()):
        self.model_dir = glob.glob(os.path.join(model_config.model_dir, f'*{model_config.model_type}*.onnx'))[0]
        sess_options = rt.SessionOptions()

        # Set graph optimization level to ORT_ENABLE_EXTENDED to enable bert optimization.
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # To enable model serialization and store the optimized graph to desired location.
        sess_options.optimized_model_filepath = self.model_dir
        self.session = rt.InferenceSession(self.model_dir, sess_options)
        self.tokenizer = Tokenizer(model_config)

    def rank(self, query: str, choices: List[str], filter_results=True):
        """
        Ranks the choices based on the similarity to the query.
        """
        if len(choices) == 0:
            return [], []

        input_ids, attention_mask, token_type_ids = self.tokenizer.encode(query, choices)

        try:
            logits = np.array(self.session.run(None, {
                'input_ids': np.array(input_ids),
                'input_mask': np.array(attention_mask),
                'segment_ids': np.array(token_type_ids)
            }))[0]
        except Exception as e:
            print(e)

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