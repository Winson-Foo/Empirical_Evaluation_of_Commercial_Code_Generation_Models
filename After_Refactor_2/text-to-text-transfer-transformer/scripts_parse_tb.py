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

"""Parse TensorBoard events files and (optionally) log results to csv.

Note that `--summary_dir` *must* point directly to the directory with .events
files (e.g. `/validation_eval/`), not a parent directory.
"""

import os

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf
from t5.evaluation import eval_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("summary_dir", None, "Where to search for .events files.")
flags.DEFINE_string("out_file", None, "Output file to write TSV.")
flags.DEFINE_bool("perplexity_eval", False,
                  "Indicates if perplexity_eval mode was used for evaluation.")
flags.DEFINE_bool("seqio_summaries", False, "Whether summaries are generated "
                  "from SeqIO Evaluator.")


def process_summary_dirs():
    """Process summary directories based on flags."""
    if FLAGS.seqio_summaries:
        subdirs = tf.io.gfile.listdir(FLAGS.summary_dir)
        summary_dirs = [os.path.join(FLAGS.summary_dir, d) for d in subdirs]
    else:
        summary_dirs = [FLAGS.summary_dir]
    return summary_dirs


def calculate_task_metrics(events, is_perplexity_eval):
    """Calculate task metrics based on events and evaluation mode."""
    if is_perplexity_eval:
        task_metrics = events
    else:
        task_metrics = eval_utils.get_eval_metric_values(
            events,
            task_name=os.path.basename(d) if FLAGS.seqio_summaries else None
        )
    return task_metrics


def run_evaluation():
    """Parse events files and log results to CSV."""
    summary_dirs = process_summary_dirs()
    scores = None
    for summary_dir in summary_dirs:
        events = eval_utils.parse_events_files(summary_dir, FLAGS.seqio_summaries)
        task_metrics = calculate_task_metrics(events, FLAGS.perplexity_eval)
        if scores:
            scores.update(task_metrics)
        else:
            scores = task_metrics

    if not scores:
        logging.info("No evaluation events found in %s", FLAGS.summary_dir)
        return

    df = eval_utils.scores_to_df(scores)
    df = eval_utils.compute_avg_glue(df)
    df = eval_utils.sort_columns(df)
    eval_utils.log_csv(df, output_file=FLAGS.out_file)


def main(_):
    run_evaluation()


if __name__ == "__main__":
    app.run(main)