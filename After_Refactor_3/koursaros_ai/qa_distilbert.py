from typing import Tuple
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import numpy as np
import torch
from nboost.plugins.qa.base import QAModelPlugin

class Tokenizer:
    @staticmethod
    def is_whitespace(c):
        if c in " \t\r\n" or ord(c) == 0x202F:
            return True
        return False

    @staticmethod
    def split_on_whitespace(choice):
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        for c in choice:
            if Tokenizer.is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        return doc_tokens, char_to_word_offset

class PtDistilBertQAModelPlugin(QAModelPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = DistilBertForQuestionAnswering.from_pretrained(self.model_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def tokenize_choice(self, choice):
        doc_tokens, char_to_word_offset = Tokenizer.split_on_whitespace(choice)
        tok_to_orig_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(doc_tokens):
            sub_tokens = self.tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
        return doc_tokens, char_to_word_offset, tok_to_orig_index, all_doc_tokens

    def get_truncated_query(self, query):
        return self.tokenizer.encode(query, add_special_tokens=False, max_length=self.max_query_length)

    def get_encoded_dict(self, truncated_query, all_doc_tokens):
        return self.tokenizer.encode_plus(truncated_query, all_doc_tokens, max_length=self.max_seq_len, return_tensors='pt')

    def get_logits(self, encoded_dict, truncated_query):
        start_logits, end_logits = self.model(input_ids=encoded_dict['input_ids'].to(self.device))
        start_logits, end_logits = start_logits.cpu(), end_logits.cpu()
        start_logits = start_logits[0][len(truncated_query) + 2:-1]
        end_logits = end_logits[0][len(truncated_query) + 2:-1]
        return start_logits, end_logits

    def get_answer(self, query: str, choice: str) -> Tuple[str, int, int, float]:
        """Return (answer, (start_pos, end_pos), score)"""
        doc_tokens, char_to_word_offset, tok_to_orig_index, all_doc_tokens = self.tokenize_choice(choice)
        truncated_query = self.get_truncated_query(query)
        encoded_dict = self.get_encoded_dict(truncated_query, all_doc_tokens)
        start_logits, end_logits = self.get_logits(encoded_dict, truncated_query)

        assert len(end_logits) == len(tok_to_orig_index) or len(end_logits) - \
               self.max_seq_len - len(truncated_query) <= 3
        if len(start_logits) == 0:
            return '', 0, 0, 0

        max_score = start_logits[0] + end_logits[-1]
        start_tok = 0
        end_tok = len(end_logits) - 1
        for i, start_logit in enumerate(start_logits[:-1]):
            end_logit_pos = np.argmax(end_logits[i+1:]) + i + 1
            score = start_logit + end_logits[end_logit_pos]
            if score > max_score:
                max_score = score
                start_tok = i
                end_tok = end_logit_pos

        answer = ' '.join(doc_tokens[tok_to_orig_index[start_tok]:tok_to_orig_index[end_tok] + 1])
        start_char_offset = char_to_word_offset.index(tok_to_orig_index[start_tok])
        end_tok_offset = tok_to_orig_index[end_tok] + 1
        end_char_offset = char_to_word_offset.index(end_tok_offset - 1) if end_tok_offset in char_to_word_offset else len(char_to_word_offset) - 1
        return answer, start_char_offset, end_char_offset, float(max_score)