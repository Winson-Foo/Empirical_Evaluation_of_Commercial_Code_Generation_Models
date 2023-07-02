import json
import os

from absl.testing import absltest
from t5.evaluation import metrics
from t5.evaluation import test_utils

_TESTDATA_PREFIX = os.path.join(os.path.dirname(__file__), "testdata")
_LARGE_TARGETS_FILE = os.path.join(_TESTDATA_PREFIX, "target_large.txt")
_LARGE_PREDICTIONS_FILE = os.path.join(_TESTDATA_PREFIX, "prediction_large.txt")
_EXPECTED_RESULTS_FILE = os.path.join(_TESTDATA_PREFIX, "expected_bootstrap_results.json")

class ScoringTest(test_utils.BaseMetricsTest):
    def setUp(self):
        super().setUp()
        self.targets = self.read_file_lines(_LARGE_TARGETS_FILE)
        self.predictions = self.read_file_lines(_LARGE_PREDICTIONS_FILE)
        self.expected_bootstrap_result = self.read_json_file(_EXPECTED_RESULTS_FILE)

    def read_file_lines(self, file_path):
        with open(file_path, "r") as f:
            return f.readlines()

    def read_json_file(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def test_rouge_variants(self):
        mean_result = metrics.rouge_mean(self.targets, self.predictions)
        self.assertDictClose(mean_result, self.expected_bootstrap_result, delta=0.5)


if __name__ == "__main__":
    absltest.main()