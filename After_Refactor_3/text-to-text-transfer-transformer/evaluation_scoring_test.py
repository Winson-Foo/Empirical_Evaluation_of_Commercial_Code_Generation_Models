# copyright header
# imports
# constants
# helper functions
# test class


'''Tests for equivalent scores between different t5.evaluation.metrics.'''

import json
import os
import unittest

from absl.testing import absltest

from t5.evaluation import metrics
from t5.evaluation import test_utils

with open(os.path.join(os.path.dirname(__file__), "testdata", "target_large.txt"), "r") as f:
    TARGETS = f.readlines()

with open(os.path.join(os.path.dirname(__file__), "testdata", "prediction_large.txt"), "r") as f:
    PREDICTIONS = f.readlines()

with open(
    os.path.join(os.path.dirname(__file__), "testdata", "expected_bootstrap_results.json"), "r"
) as f:
    EXPECTED_BOOTSTRAP_RESULT = json.load(f)


class MetricsTest(test_utils.BaseMetricsTest):

    def test_rouge_variants(self):
        mean_result = metrics.rouge_mean(TARGETS, PREDICTIONS)
        self.assertDictEqual(mean_result, EXPECTED_BOOTSTRAP_RESULT, delta=0.5)


if __name__ == "__main__":
    absltest.main()