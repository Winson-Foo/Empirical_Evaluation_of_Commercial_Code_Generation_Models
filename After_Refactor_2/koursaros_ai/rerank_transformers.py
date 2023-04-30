from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List
import torch.nn
import torch
from nboost.plugins.rerank.base import RerankModelPlugin
from nboost import defaults
from nboost.logger import set_logger


class PtTransformersRerankPlugin(RerankModelPlugin):
    """Reranker models based on huggingface/transformers library"""

    def __init__(self,
                 model_dir: str = 'nboost/pt-tinybert-msmarco',
                 verbose: bool = defaults.verbose,
                 max_seq_len: int = defaults.max_seq_len,
                 **kwargs):
        """
        Initialize the reranking plugin with a pre-trained transformers model.

        :param model_dir: the directory containing the model files
        :param verbose: whether to output debugging information
        :param max_seq_len: the maximum length of the input sequence
        """
        super().__init__(**kwargs)
        self.logger = set_logger(model_dir, verbose=verbose)
        self.max_seq_len = max_seq_len

        self.logger.info('Loading from checkpoint %s' % model_dir)

        # Use the appropriate device based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Output device information to logger
        if self.device == torch.device("cpu"):
            self.logger.info("RUNNING ON CPU")
        else:
            self.logger.info("RUNNING ON CUDA")
            torch.cuda.synchronize(self.device)

        # Load the pre-trained model and tokenizer
        with torch.no_grad():
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device, non_blocking=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    def get_logits(self, query: str, choices: List[str]) -> List[float]:
        """
        Compute the logits for a given query and list of choices.

        :param query: the query string
        :param choices: a list of choice strings
        :return: a list of logits (one for each choice)
        """
        input_ids, attention_mask, token_type_ids = self.encode(query, choices)

        with torch.no_grad():
            logits = self.rerank_model(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)[0]
            logits = logits.detach().cpu().numpy().tolist()

            return logits

    def encode(self, query: str, choices: List[str]) -> torch.Tensor:
        """
        Encode a given query and list of choices as input tensors for the pre-trained model.

        :param query: the query string
        :param choices: a list of choice strings
        :return: the input tensors
        """
        # Encode each choice with the query and a special token
        encoded_choices = [self.tokenizer.encode_plus(
            query, choice, add_special_tokens=True, return_token_type_ids=True, max_length=self.max_seq_len
            ) for choice in choices]

        # Find the maximum input length (to avoid padding unnecessary tokens)
        max_input_length = min(max(len(t['input_ids']) for t in encoded_choices), self.max_seq_len)

        # Pad the input sequences to the same length and create attention masks and token type IDs
        input_ids = [t['input_ids'][:max_input_length] +
                     [0] * (max_input_length - len(t['input_ids'][:max_input_length])) for t in encoded_choices]
        attention_mask = [[1] * len(t['input_ids'][:max_input_length]) +
                          [0] * (max_input_length - len(t['input_ids'][:max_input_length])) for t in encoded_choices]
        token_type_ids = [t['token_type_ids'][:max_input_length] +
                          [0] * (max_input_length - len(t['token_type_ids'][:max_input_length])) for t in encoded_choices]

        # Convert the input tensors to PyTorch tensors and move them to the appropriate device
        input_ids = torch.tensor(input_ids).to(self.device, non_blocking=True)
        attention_mask = torch.tensor(attention_mask).to(self.device, non_blocking=True)
        token_type_ids = torch.tensor(token_type_ids).to(self.device, non_blocking=True)

        return input_ids, attention_mask, token_type_ids