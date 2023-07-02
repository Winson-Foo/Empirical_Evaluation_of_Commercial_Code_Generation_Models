import collections
import re
import string

from absl import logging
import numpy as np


def normalize_answer(text):
    """Normalize the answer by lowercasing and removing punctuation, articles and extra whitespace."""
    PUNC_CHARS = string.punctuation + "‘’´`_"
    PUNC_REPL = " "
  
    # Lowercase the text
    text = text.lower()
    
    # Replace punctuation characters with the replacement string
    to_replace = set(PUNC_CHARS)
    text = "".join(PUNC_REPL if ch in to_replace else ch for ch in text)
    
    # Remove articles (a, an, the)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    
    # Remove extra whitespaces
    text = " ".join(text.split())
    
    return text.strip()


def normalize_triviaqa_answer(answer):
    """Normalization used in official TriviaQA evaluation script."""
    return normalize_answer(answer)


def normalize_squad_answer(answer):
    """Normalization used in official SQuAD evaluation script."""
    return normalize_answer(answer)


def compute_max_metric_over_ground_truths(metric_fn, ground_truths, prediction):
    """Compute the maximum of the metric over all ground truths."""
    return max(metric_fn(ground_truth, prediction) for ground_truth in ground_truths)


def compute_exact_match_score(target, prediction):
    return target == prediction


def compute_f1_score(target, prediction):
    """Compute token f1 score for a single target and prediction."""
    prediction_tokens = prediction.split()
    target_tokens = target.split()
    common = (collections.Counter(prediction_tokens) & collections.Counter(target_tokens))
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(target_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_qa_metrics(targets, predictions):
    """Compute exact match and f1 QA scores, expecting pre-normalized text."""
    if len(targets) != len(predictions):
        raise ValueError("Number of targets and predictions must match.")
    
    em = np.mean([
        compute_max_metric_over_ground_truths(compute_exact_match_score, t, p)
        for p, t in zip(predictions, targets)
    ])
    
    f1 = np.mean([
        compute_max_metric_over_ground_truths(compute_f1_score, t, p)
        for p, t in zip(predictions, targets)
    ])
    
    em *= 100
    f1 *= 100
    
    logging.info("EM = %.2f, F1 = %.2f", em, f1)
    
    return {"em": em, "f1": f1}