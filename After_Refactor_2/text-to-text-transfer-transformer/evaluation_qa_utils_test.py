# Copyright 2023 The T5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for t5.evaluation.qa_utils."""

from typing import List, Dict
from absl.testing import absltest
from t5.evaluation import qa_utils


class QaUtilsTest(absltest.TestCase):

    def test_normalize_trivia_qa(self):
        input_text = "`Needs\tA_LOT of the 'normalization'.\"‘"
        expected_output = "needs lot of normalization"
        self.assertEqual(qa_utils.normalize_trivia_qa(input_text), expected_output)

        input_text = "needs no normalization"
        expected_output = "needs no normalization"
        self.assertEqual(qa_utils.normalize_trivia_qa(input_text), expected_output)

    def test_normalize_squad(self):
        input_text = "`Needs\tA_LOT of the 'normalization'.\"‘"
        expected_output = "needs alot of normalization‘"
        self.assertEqual(qa_utils.normalize_squad(input_text), expected_output)

        input_text = "needs no normalization"
        expected_output = "needs no normalization"
        self.assertEqual(qa_utils.normalize_squad(input_text), expected_output)

    def test_qa_metrics(self):
        targets: List[List[str]] = [["answer"]] * 5
        predictions: List[str] = ["answer"] * 5
        expected_output: Dict[str, float] = {"em": 100.0, "f1": 100.0}
        self.assertDictEqual(qa_utils.qa_metrics(targets, predictions), expected_output)

        targets = [
            ["big moose", "hippo"],
            ["correct1"],
            ["correct2.1", "correct2.2"],
            ["a", "b"],
        ]
        predictions = [
            "a big moose‘",
            "wrong",
            "correct2.2",
            "c",
        ]
        expected_output = {"em": 25.0, "f1": 35.0}
        self.assertDictEqual(qa_utils.qa_metrics(targets, predictions), expected_output)


if __name__ == "__main__":
    absltest.main()