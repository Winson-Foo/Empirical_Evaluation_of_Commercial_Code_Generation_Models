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

import collections
import os
import numpy as np
import pandas as pd
import seqio
import tensorflow.compat.v1 as tf
from absl.testing import absltest

from t5.evaluation import eval_utils


class EvalUtilsTest(absltest.TestCase):
    
    def setUp(self):
        self.tb_summary_dir = self.create_tempdir()

    def tearDown(self):
        del self.tb_summary_dir

    def test_parse_events_files(self):
        with tf.Graph().as_default():
            summary_writer = tf.summary.FileWriter(self.tb_summary_dir.full_path)
            tags = ["eval/foo_task/accuracy",
                    "eval/foo_task/accuracy",
                    "loss"
            ]
            values = [1., 2., 3.]
            steps = [20, 30, 40]
            summary = tf.Summary()

            for tag, value, step in zip(tags, values, steps):
                summary.value.add(tag=tag, simple_value=value)
                summary_writer.add_summary(summary, step)

            summary_writer.flush()

        events = eval_utils.parse_events_files(self.tb_summary_dir.full_path)
        expected_events = {
            "eval/foo_task/accuracy": [(20, 1.), (30, 2.)],
            "loss": [(40, 3.)],
        }

        self.assertDictEqual(events, expected_events)

    def test_parse_events_files_seqio(self):
        metrics = [{"accuracy": seqio.metrics.Scalar(1.)},
                   {"accuracy": seqio.metrics.Scalar(2.)}]
        steps = [20, 30]

        logger = seqio.TensorBoardLoggerV1(self.tb_summary_dir.full_path)
        for metric, step in zip(metrics, steps):
            logger(task_name="foo_task", metrics=metric, step=step, dataset=tf.data.Dataset.range(0), inferences={}, targets=[])

        events = eval_utils.parse_events_files(os.path.join(self.tb_summary_dir.full_path, "foo_task"), seqio_summaries=True)

        expected_events = {
            "eval/accuracy": [(20, 1.), (30, 2.)],
        }

        self.assertDictEqual(events, expected_events)

    def test_get_eval_metric_values(self):
        events = {
            "eval/foo_task/accuracy": [(20, 1.), (30, 2.)],
            "eval/bar_task/sequence_accuracy": [(10, 3.)],
            "loss": [(40, 3.)],
        }
        eval_values = eval_utils.get_eval_metric_values(events)
        expected_eval_values = {
            "foo_task/accuracy": [(20, 1.), (30, 2.)],
            "bar_task/sequence_accuracy": [(10, 3.)],
        }

        self.assertDictEqual(eval_values, expected_eval_values)

    def test_get_eval_metric_values_seqio(self):
        events = {
            "eval/accuracy": [(20, 1.), (30, 2.)],
            "eval/sequence_accuracy": [(10, 3.)],
            "loss": [(40, 3.)],
        }
        eval_values = eval_utils.get_eval_metric_values(events, task_name="foo_task")
        expected_eval_values = {
            "foo_task/accuracy": [(20, 1.), (30, 2.)],
            "foo_task/sequence_accuracy": [(10, 3.)],
        }

        self.assertDictEqual(eval_values, expected_eval_values)

    def test_glue_average(self):
        df = pd.DataFrame(columns=["ABC Accuracy", "DEF Exact Match", "DEF Accuracy"])

        df.loc[10] = [0, 0, 4]
        df.loc[20] = [1, 10, 7]
        df.loc[30] = [2, 3, 8]
        df.loc[40] = [3, 0, 0]

        df = eval_utils.compute_avg_glue(df)

        self.assertAlmostEqual(df["Average GLUE Score"][10], 0.0)
        self.assertAlmostEqual(df["Average GLUE Score"][20], 4.75)
        self.assertAlmostEqual(df["Average GLUE Score"][30], 5.75)
        self.assertAlmostEqual(df["Average GLUE Score"][40], 6.0)
        self.assertAlmostEqual(df["Average SuperGLUE Score"][10], 4.0)
        self.assertAlmostEqual(df["Average SuperGLUE Score"][20], 8.0)
        self.assertAlmostEqual(df["Average SuperGLUE Score"][30], 11.5)
        self.assertAlmostEqual(df["Average SuperGLUE Score"][40], 11.5)

    def test_metric_group_max(self):
        df = pd.DataFrame(columns=["ABC Accuracy", "DEF Exact Match", "DEF Accuracy"])
        df.loc[10] = [1., 0., 4.]
        df.loc[20] = [2., 10., 7.]
        df.loc[30] = [3., 3., 8.]
        df.loc[40] = [4., 0., 0.]

        metric_names = collections.OrderedDict([
            ("metric1", eval_utils.Metric("ABC Accuracy")),
            ("metric2", eval_utils.Metric("DEF Accuracy", "DEF")),
            ("metric3", eval_utils.Metric("DEF Exact Match", "DEF")),
        ])

        metric_max, metric_max_step = eval_utils.metric_group_max(df, metric_names)
        expected_metric_max = {
            "ABC Accuracy": 4.0,
            "DEF Accuracy": 10.0,
            "DEF Exact Match": 8.0,
        }
        expected_metric_max_step = {
            "ABC Accuracy": 40,
            "DEF Accuracy": 20,
            "DEF Exact Match": 30,
        }

        self.assertDictEqual(metric_max, expected_metric_max)
        self.assertDictEqual(metric_max_step, expected_metric_max_step)

    def test_log_csv(self):
        metric_names = list(eval_utils.METRIC_NAMES.values())

        df = pd.DataFrame(columns=[m.name for m in metric_names[:3]])
        df.loc[10] = [np.nan, 1., 2.]
        df.loc[20] = [3., np.nan, np.nan]
        df.loc[30] = [4., np.nan, np.nan]

        df.index.name = "step"
        output_file = os.path.join(self.tb_summary_dir.full_path, "results.csv")

        eval_utils.log_csv(df, output_file=output_file)

        with tf.io.gfile.GFile(output_file) as f:
            output = f.read()

        expected_output = """step,{},{},{}
10,,1.0,2.0
20,3.0,,
30,4.0,,
max,4.0,2.0,3.0
step,30,10,10""".format(*[m.name for m in metric_names[:3]])

        self.assertEqual(output, expected_output)


if __name__ == "__main__":
    absltest.main()