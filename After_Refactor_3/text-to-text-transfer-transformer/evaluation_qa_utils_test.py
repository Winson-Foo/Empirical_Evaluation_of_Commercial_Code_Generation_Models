"""Tests for t5.evaluation.qa_utils."""

from absl.testing import absltest
from t5.evaluation import qa_utils


class QaUtilsTest(absltest.TestCase):

    def test_normalize_trivia_qa(self):
        self.assertEqual(
            qa_utils.normalize_trivia_qa("`Needs\tA_LOT of the 'normalization'.\"‘"),
            "needs lot of normalization",
        )
        self.assertEqual(
            qa_utils.normalize_trivia_qa("needs no normalization"),
            "needs no normalization",
        )

    def test_normalize_squad(self):
        self.assertEqual(
            qa_utils.normalize_squad("`Needs\tA_LOT of the 'normalization'.\"‘"),
            "needs alot of normalization‘",
        )
        self.assertEqual(
            qa_utils.normalize_squad("needs no normalization"),
            "needs no normalization",
        )

    def test_qa_metrics(self):
        with self.assertRaisesRegex(ValueError, "Number of targets and predictions must match."):
            qa_utils.qa_metrics([["answer"]] * 6, ["answer"] * 5)

        self.assertDictEqual(
            qa_utils.qa_metrics([["answer"]] * 5, ["answer"] * 5),
            {"em": 100.0, "f1": 100.0}
        )

        self.assertDictEqual(
            qa_utils.qa_metrics(
                [["big moose", "hippo"],
                 ["correct1"],
                 ["correct2.1", "correct2.2"],
                 ["a", "b"]],
                ["a big moose‘",
                 "wrong",
                 "correct2.2",
                 "c"],
            ),
            {"em": 25., "f1": 35.},
        )


if __name__ == "__main__":
    absltest.main()