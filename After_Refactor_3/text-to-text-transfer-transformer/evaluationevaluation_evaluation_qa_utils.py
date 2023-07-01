import collections
import re
import string

from absl import logging
import numpy as np


def normalize_answer(answer):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
        punc_chars = string.punctuation
        punc_repl = ""
        return "".join(punc_repl if ch in punc_chars else ch for ch in s)

    def white_space_fix(s):
        return " ".join(s.split())

    text = answer.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text


def qa_metrics(targets, predictions):
    """Computes exact match and f1 QA scores, expecting pre-normalized text."""
    if len(targets) != len(predictions):
        raise ValueError("Number of targets and predictions must match.")
    
    def metric_max_over_ground_truths(metric_fn, ground_truths, prediction):
        """Computes the maximum of the metric over all ground truths."""
        return max(metric_fn(ground_truth, prediction) for ground_truth in ground_truths)

    def exact_match_score(target, prediction):
        return target == prediction

    def f1_score(target, prediction):
        """Computes token f1 score for a single target and prediction."""
        prediction_tokens = prediction.split()
        target_tokens = target.split()
        common = (collections.Counter(prediction_tokens) &
                  collections.Counter(target_tokens))
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(target_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    em = np.mean([
        metric_max_over_ground_truths(exact_match_score, t, p)
        for p, t in zip(predictions, targets)
    ])
    f1 = np.mean([
        metric_max_over_ground_truths(f1_score, t, p)
        for p, t in zip(predictions, targets)
    ])
    em *= 100
    f1 *= 100
    logging.info("EM = %.2f, F1 = %.2f", em, f1)
    return {"em": em, "f1": f1}