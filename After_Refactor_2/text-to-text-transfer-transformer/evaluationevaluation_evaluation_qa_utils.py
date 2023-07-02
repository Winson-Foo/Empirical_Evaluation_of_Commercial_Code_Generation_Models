import re
import string
import numpy as np

class QAEvaluation:
  def __init__(self):
    self.punc_chars = string.punctuation
    self.punc_repl = ""

  def normalize_answer(self, text):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(s):
      return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
      to_replace = set(self.punc_chars)
      return "".join(self.punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
      return " ".join(s.split())

    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text

  def normalize_trivia_qa(self, answer):
    """Normalization used in official TriviaQA evaluation script."""
    return self.normalize_answer(answer).strip()

  def normalize_squad(self, answer):
    """Normalization used in official SQuAD evaluation script."""
    return self.normalize_answer(answer)

  def metric_max_over_ground_truths(self, metric_fn, ground_truths, prediction):
    """Computes the maximum of the metric over all ground truths."""
    return max(
      metric_fn(ground_truth, prediction) for ground_truth in ground_truths
    )

  def exact_match_score(self, target, prediction):
    return target == prediction

  def f1_score(self, target, prediction):
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

  def qa_metrics(self, targets, predictions):
    """Computes exact match and f1 QA scores, expecting pre-normalized text."""
    if len(targets) != len(predictions):
      raise ValueError("Number of targets and predictions must match.")
    em = np.mean([
      self.metric_max_over_ground_truths(self.exact_match_score, t, p)
      for p, t in zip(predictions, targets)
    ])
    f1 = np.mean([
      self.metric_max_over_ground_truths(self.f1_score, t, p)
      for p, t in zip(predictions, targets)
    ])
    em *= 100
    f1 *= 100
    logging.info("EM = %.2f, F1 = %.2f", em, f1)
    return {"em": em, "f1": f1}