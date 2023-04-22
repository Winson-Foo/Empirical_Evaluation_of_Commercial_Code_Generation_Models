import glob
import os
from typing import List

import numpy as np
import onnxruntime as rt
from transformers import AutoTokenizer

from nboost import defaults
from nboost.plugins.models.rerank.base import RerankModelPlugin

MAX_SEQ_LEN = 512


class ONNXBertRerankModelPlugin(RerankModelPlugin):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_model()

    def rank(self, query: str, choices: List[str], filter_results=defaults.filter_results):
        """
        Rank choices based on a given query using ONNX Bert model.
        :param query: A string specifying the query.
        :param choices: A list of strings specifying the choices to be ranked.
        :param filter_results: A boolean to filter results based on score.
        :return: A tuple containing the sorted indices of the choices and their corresponding scores.
        """
        if len(choices) == 0:
            return [], []

        input_ids, attention_mask, token_type_ids = self._encode_inputs(query, choices)

        logits = np.array(self.session.run(None, {
            'input_ids': np.array(input_ids),
            'input_mask': np.array(attention_mask),
            'segment_ids': np.array(token_type_ids)
        }))[0]

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

    def _load_model(self):
        model_dir = glob.glob(os.path.join(self.model_dir, '*.onnx'))[0]
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        sess_options.optimized_model_filepath = model_dir
        self.session = rt.InferenceSession(model_dir, sess_options)
        self.tokenizer = AutoTokenizer.from_pretrained('albert-base-uncased' if 'albert' in model_dir else 'bert-base-uncased')

    def _encode_inputs(self, query: str, choices: List[str]):
        inputs = [self.tokenizer.encode_plus(query, choice, add_special_tokens=True)
                  for choice in choices]

        max_len = min(max(len(t['input_ids']) for t in inputs), MAX_SEQ_LEN)
        input_ids = [t['input_ids'][:max_len] + [0] * (max_len - len(t['input_ids'][:max_len])) for t in inputs]
        attention_mask = [[1] * len(t['input_ids'][:max_len]) + [0] * (max_len - len(t['input_ids'][:max_len])) for t in
                          inputs]
        token_type_ids = [t['token_type_ids'][:max_len] + [0] * (max_len - len(t['token_type_ids'][:max_len])) for t in
                          inputs]

        return input_ids, attention_mask, token_type_ids
