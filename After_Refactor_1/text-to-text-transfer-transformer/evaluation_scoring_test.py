"""Tests for equivalent scores between different t5.evaluation.metrics."""

import json
import os
import unittest

from absl.testing import absltest

from t5.evaluation import metrics
from t5.evaluation import test_utils

# Delta for matching rouge values between the different scorers.
ROUGE_DELTA = 0.5

# File paths
TESTDATA_PREFIX = os.path.join(os.path.dirname(__file__), "testdata")
LARGE_TARGETS_FILE = os.path.join(TESTDATA_PREFIX, "target_large.txt")
LARGE_PREDICTIONS_FILE = os.path.join(TESTDATA_PREFIX, "prediction_large.txt")
EXPECTED_RESULTS_FILE = os.path.join(TESTDATA_PREFIX, "expected_bootstrap_results.json")


class ScoringTest(test_utils.BaseMetricsTest):
    """Tests for scoring metrics."""

    def setUp(self):
        super().setUp()
        self.targets = self._read_lines_from_file(LARGE_TARGETS_FILE)
        self.predictions = self._read_lines_from_file(LARGE_PREDICTIONS_FILE)
        self.expected_bootstrap_result = self._read_json_from_file(EXPECTED_RESULTS_FILE)

    def test_rouge_variants(self):
        """Test ROUGE variants."""
        mean_result = metrics.rouge_mean(self.targets, self.predictions)
        self.assertDictClose(mean_result, self.expected_bootstrap_result, delta=ROUGE_DELTA)

    @staticmethod
    def _read_lines_from_file(file_path):
        """Read lines from a file and return as a list."""
        with open(file_path, "r") as file:
            return file.readlines()

    @staticmethod
    def _read_json_from_file(file_path):
        """Read JSON data from a file and return as a dictionary."""
        with open(file_path, "r") as file:
            return json.load(file)


if __name__ == "__main__":
    absltest.main()