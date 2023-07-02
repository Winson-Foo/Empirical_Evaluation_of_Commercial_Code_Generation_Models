from absl.testing import absltest
import pandas as pd

from t5.evaluation import eval_utils


class EvalUtilsTest(absltest.TestCase):

    def test_parse_events_files(self):
        summary_dir = self.create_tempdir()
        tb_summary_dir = os.path.join(summary_dir, "tb")

        with tf.Graph().as_default():
            summary_writer = tf.summary.FileWriter(tb_summary_dir)
            tags = ["eval/foo_task/accuracy", "eval/foo_task/accuracy", "loss"]
            values = [1.0, 2.0, 3.0]
            steps = [20, 30, 40]

            for tag, value, step in zip(tags, values, steps):
                summary = tf.Summary()
                summary.value.add(tag=tag, simple_value=value)
                summary_writer.add_summary(summary, step)

            summary_writer.flush()

        events = eval_utils.parse_events_files(tb_summary_dir)
        expected_events = {
            "eval/foo_task/accuracy": [(20, 1.0), (30, 2.0)],
            "loss": [(40, 3.0)]
        }

        self.assertDictEqual(events, expected_events)

    def test_get_eval_metric_values(self):
        events = {
            "eval/foo_task/accuracy": [(20, 1.0), (30, 2.0)],
            "eval/bar_task/sequence_accuracy": [(10, 3.0)],
            "loss": [(40, 3.0)],
        }

        eval_values = eval_utils.get_eval_metric_values(events)
        expected_eval_values = {
            "foo_task/accuracy": [(20, 1.0), (30, 2.0)],
            "bar_task/sequence_accuracy": [(10, 3.0)],
        }

        self.assertDictEqual(eval_values, expected_eval_values)

    def test_compute_avg_glue(self):
        columns = [
            "glue_metric1",
            "glue_metric2",
            "super_metric1",
            "super_metric2",
            "Extra metric",
            "Average GLUE Score",
            "Average SuperGLUE Score",
        ]

        df = pd.DataFrame(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            ],
            columns=columns
        )

        avg_glue_df = eval_utils.compute_avg_glue(df)

        expected_avg_glue_values = [
            (0 + 1) / 2.0,
            (2 + 3) / 2.0,
            (4 + 5) / 2.0,
            (6 + 7) / 2.0
        ]

        self.assertListAlmostEqual(avg_glue_df["Average GLUE Score"], expected_avg_glue_values)

    def test_metric_group_max(self):
        df = pd.DataFrame(
            {
                "ABC Accuracy": [1.0, 2.0, 3.0, 4.0],
                "DEF Exact Match": [0.0, 10.0, 3.0, 0.0],
                "DEF Accuracy": [4.0, 7.0, 8.0, 0.0],
            },
            index=[10, 20, 30, 40],
        )

        metric_names = {
            "metric1": eval_utils.Metric("ABC Accuracy"),
            "metric2": eval_utils.Metric("DEF Accuracy", "DEF"),
            "metric3": eval_utils.Metric("DEF Exact Match", "DEF"),
        }

        metric_max, metric_max_step = eval_utils.metric_group_max(df, metric_names)

        expected_metric_max = {
            "ABC Accuracy": 4.0,
            "DEF Accuracy": 8.0,
            "DEF Exact Match": 10.0,
        }

        expected_metric_max_step = {
            "ABC Accuracy": 40,
            "DEF Accuracy": 30,
            "DEF Exact Match": 20,
        }

        self.assertDictEqual(metric_max, expected_metric_max)
        self.assertDictEqual(metric_max_step, expected_metric_max_step)


if __name__ == "__main__":
    absltest.main()