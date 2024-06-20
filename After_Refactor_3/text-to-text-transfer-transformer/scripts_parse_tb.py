"""Parse TensorBoard events files and (optionally) log results to csv.

Note that `--summary_dir` *must* point directly to the directory with .events
files (e.g. `/validation_eval/`), not a parent directory.
"""

import os
from typing import Dict, Union

from absl import app, flags, logging
from t5.evaluation import eval_utils
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string("summary_dir", None, "Where to search for .events files.")
flags.DEFINE_string("out_file", None, "Output file to write TSV.")
flags.DEFINE_bool("perplexity_eval", False, "Indicates if perplexity_eval mode was used for evaluation.")
flags.DEFINE_bool("seqio_summaries", False, "Whether summaries are generated from SeqIO Evaluator.")


class TensorBoardParser:
    def __init__(self, summary_dir: str, out_file: str, perplexity_eval: bool, seqio_summaries: bool):
        self.summary_dir = summary_dir
        self.out_file = out_file
        self.perplexity_eval = perplexity_eval
        self.seqio_summaries = seqio_summaries

    def parse_events(self) -> Union[None, Dict[str, Dict[str, float]]]:
        """Parse TensorBoard events and return task metrics."""
        if self.seqio_summaries:
            subdirs = tf.io.gfile.listdir(self.summary_dir)
            summary_dirs = [os.path.join(self.summary_dir, d) for d in subdirs]
        else:
            summary_dirs = [self.summary_dir]

        task_metrics = None
        for directory in summary_dirs:
            events = eval_utils.parse_events_files(directory, self.seqio_summaries)
            if self.perplexity_eval:
                metrics = events
            else:
                metrics = eval_utils.get_eval_metric_values(events, task_name=os.path.basename(directory) if self.seqio_summaries else None)
            if task_metrics:
                task_metrics.update(metrics)
            else:
                task_metrics = metrics

        return task_metrics

    @staticmethod
    def log_csv(scores: Dict[str, Dict[str, float]], output_file: str) -> None:
        """Log task scores to CSV file."""
        df = eval_utils.scores_to_df(scores)
        df = eval_utils.compute_avg_glue(df)
        df = eval_utils.sort_columns(df)
        eval_utils.log_csv(df, output_file=output_file)


def main(_):
    parser = TensorBoardParser(FLAGS.summary_dir, FLAGS.out_file, FLAGS.perplexity_eval, FLAGS.seqio_summaries)
    task_metrics = parser.parse_events()

    if not task_metrics:
        logging.info("No evaluation events found in %s", FLAGS.summary_dir)
        return

    parser.log_csv(task_metrics, output_file=FLAGS.out_file)


if __name__ == "__main__":
    app.run(main)