from transformers import AutoTokenizer
from config import ModelConfig

class Tokenizer:
    """
    Wrapper for AutoTokenizer from transformers.
    """

    def __init__(self, model_config=ModelConfig()):
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    def encode(self, query, choices):
        """
        Encodes the input using the tokenizer.
        """
        inputs = [self.tokenizer.encode_plus(query, choice, add_special_tokens=True)
                  for choice in choices]

        max_len = min(max(len(t['input_ids']) for t in inputs), self.max_seq_len)
        input_ids = [t['input_ids'][:max_len] +
                     [0] * (max_len - len(t['input_ids'][:max_len])) for t in inputs]
        attention_mask = [[1] * len(t['input_ids'][:max_len]) +
                          [0] * (max_len - len(t['input_ids'][:max_len])) for t in inputs]
        token_type_ids = [t['token_type_ids'][:max_len] +
                          [0] * (max_len - len(t['token_type_ids'][:max_len])) for t in inputs]

        return input_ids, attention_mask, token_type_ids