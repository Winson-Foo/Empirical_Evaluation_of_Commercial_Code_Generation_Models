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

"""Prepare a file for submission to the (Super)GLUE leaderboard.

This script assumes that you have already generated predictions on a given
split. The predictions should be saved line-by-line in a text file, and should
be written with postprocessing applied. This is the format that gets written out
when you run the Mesh TensorFlow Transformer in eval mode. Note that the order
of this line must exactly match the order of examples returned by loading the
splits from the task. So, for example, you should run eval on the cached test
set and then run this script with the split flag set to test and the cached flag
set to True.
"""

import ast
import collections
import csv
import json
import os

from absl import app
from absl import flags
import t5.data
import t5.data.tasks

flags.DEFINE_string("predictions_file", None, "Path to model predictions.")
flags.DEFINE_string("task", None, "T5 task name for this benchmark.")
flags.DEFINE_string("tfds_name", None, "Short name of tfds (e.g. 'cb').")
flags.DEFINE_string("out_dir", None, "Path to write output file.")
flags.DEFINE_string("split", "test", "Split, should typically be test.")
flags.DEFINE_boolean("super", False, "Whether to make SuperGLUE-style file.")
flags.DEFINE_boolean("cached", True, "Whether to used cached dataset.")
flags.DEFINE_list("additional_task_cache_dirs", [], "Dirs with cached tasks.")

def read_prediction_lines(file_path):
    with tf.io.gfile.GFile(file_path) as f:
        prediction_lines = f.readlines()
    return [l.strip() for l in prediction_lines]

def convert_predictions(predictions):
    if FLAGS.tfds_name in USES_TEXT:
        if FLAGS.super:
            builder_configs = tfds.text.super_glue.SuperGlue.builder_configs
        else:
            builder_configs = tfds.text.glue.Glue.builder_configs
        label_classes = builder_configs[FLAGS.tfds_name].label_classes
        return [label_classes[p] for p in predictions]
    elif FLAGS.tfds_name in ["boolq", "wic"]:
        return [("false", "true")[p] for p in predictions]
    elif FLAGS.tfds_name == "wsc":
        return [("False", "True")[p] for p in predictions]
    elif FLAGS.tfds_name == "multirc":
        rows = collections.defaultdict(lambda: collections.defaultdict(dict))
        predictions = [int(p["value"]) for p in predictions]
        for p, e in zip(predictions, examples):
            e = {k: int(e["idx/" + k]) for k in ["paragraph", "question", "answer"]}
            rows[e["paragraph"]][e["question"]][e["answer"]] = p
        return rows
    else:
        return predictions

def write_predictions(out_file, indices, predictions):
    if FLAGS.super:
        lines = [json.dumps({"idx": int(i), "label": p}) + os.linesep for i, p in zip(indices, predictions)]
        with tf.io.gfile.GFile(out_file.format(extension="jsonl"), "w") as f:
            f.writelines(lines)
    else:
        with tf.io.gfile.GFile(out_file.format(extension="tsv"), "w") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow(["index", "prediction"])
            tsv_writer.writerows([[i, p] for i, p in zip(indices, predictions)])

def main(_):
    t5.data.add_global_cache_dirs(FLAGS.additional_task_cache_dirs)
    out_file = os.path.join(FLAGS.out_dir, "{}.{{extension}}".format(TFDS_NAME_MAP[FLAGS.tfds_name]))

    ds = t5.data.TaskRegistry.get_dataset(FLAGS.task, _FAKE_LEN, FLAGS.split, use_cached=FLAGS.cached, shuffle=False)
    examples = [{k: v.numpy() for k, v in ex.items()} for ex in ds]

    prediction_lines = read_prediction_lines(FLAGS.predictions_file)
    predictions = convert_predictions([ast.literal_eval(l) for l in prediction_lines])

    if len(predictions) != len(examples):
        raise ValueError("Number of predictions in {} ({}) != of examples in {} split of {} "
                         "({}).".format(FLAGS.predictions_file,
                                        len(predictions),
                                        FLAGS.split,
                                        FLAGS.task,
                                        len(examples)))

    if "record" in FLAGS.task:
        indices = [ex["idx/query"] for ex in examples]
    else:
        indices = [ex.get("idx", None) for ex in examples]

    write_predictions(out_file, indices, predictions)

if __name__ == "__main__":
    app.run(main)