from typing import Tuple
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import numpy as np
import torch
from nboost.plugins.qa.base import QAModelPlugin


def is_whitespace(c: str) -> bool:
    """Check if character is whitespace."""
    if c in " \t\r\n" or ord(c) == 0x202F:
        return True
    return False


class PtDistilBertQAModelPlugin(QAModelPlugin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = DistilBertForQuestionAnswering.from_pretrained(self.model_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_answer(self, query: str, choice: str) -> Tuple[str, int, int, float]:
        """Return (answer, (start_pos, end_pos), score)"""
        doc_tokens, char_to_word_offset = tokenize_choice(choice)

        truncated_query = self.tokenizer.encode(query,
                                                add_special_tokens=False,
                                                max_length=self.max_query_length)

        encoded_dict = encode_input(self.tokenizer, truncated_query, doc_tokens,
                                    self.max_seq_len, self.device)

        self.model.eval()
        with torch.no_grad():
            start_logits, end_logits = self.model(
                input_ids=encoded_dict['input_ids'].to(self.device))
            start_logits, end_logits = start_logits.cpu(), end_logits.cpu()
            start_logits = start_logits[0][len(truncated_query) + 2:-1]
            end_logits = end_logits[0][len(truncated_query) + 2:-1]

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

        answer, start_char_offset, end_char_offset = get_answer_info(
            doc_tokens, char_to_word_offset, start_tok, end_tok)

        return answer, start_char_offset, end_char_offset, float(max_score)


def tokenize_choice(choice: str) -> Tuple[list, list]:
    """Return tokenized choice and character to word offset mapping."""
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True

    for c in choice:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    return doc_tokens, char_to_word_offset


def encode_input(tokenizer, truncated_query, doc_tokens, max_seq_len, device):
    """Return encoded input with attention masks."""
    tok_to_orig_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(doc_tokens):
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    encoded_dict = tokenizer.encode_plus(
        truncated_query,
        all_doc_tokens,
        max_length=max_seq_len,
        return_tensors='pt'
    )

    return encoded_dict


def get_answer_info(doc_tokens, char_to_word_offset, start_tok, end_tok):
    """Return answer text and character offsets."""
    start_char_offset = char_to_word_offset.index(
        tok_to_orig_index[start_tok])
    end_tok_offset = tok_to_orig_index[end_tok] + 1
    if end_tok_offset in char_to_word_offset:
        end_char_offset = char_to_word_offset.index(end_tok_offset) - 1
    else:
        end_char_offset = len(char_to_word_offset) - 1
    answer = ' '.join(doc_tokens[
                      tok_to_orig_index[start_tok]
                      :tok_to_orig_index[end_tok] + 1])

    return answer, start_char_offset, end_char_offset