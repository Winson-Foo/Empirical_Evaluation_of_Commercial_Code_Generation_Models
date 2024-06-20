# Import necessary modules
import ast
import collections
import csv
import json
import os

from absl import app, flags
import t5.data
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

# Define flags
flags.DEFINE_string("predictions_file", None, "Path to model predictions.")
flags.DEFINE_string("task", None, "T5 task name for this benchmark.")
flags.DEFINE_string("tfds_name", None, "Short name of tfds (e.g. 'cb').")
flags.DEFINE_string("out_dir", None, "Path to write output file.")
flags.DEFINE_string("split", "test", "Split, should typically be test.")
flags.DEFINE_boolean("super", False, "Whether to make SuperGLUE-style file.")
flags.DEFINE_boolean("cached", True, "Whether to used cached dataset.")
flags.DEFINE_list("additional_task_cache_dirs", [], "Dirs with cached tasks.")

# Mapping between tfds_name and file names
FILE_NAME_MAP = {
    "boolq": "BoolQ",
    "cb": "CB",
    "copa": "COPA",
    "multirc": "MultiRC",
    "record": "ReCoRD",
    "rte": "RTE",
    "wic": "WiC",
    "cola": "CoLA",
    "sst2": "SST-2",
    "mrpc": "MRPC",
    "stsb": "STS-B",
    "qqp": "QQP",
    "mnli_matched": "MNLI-m",
    "mnli_mismatched": "MNLI-mm",
    "qnli": "QNLI",
    "wnli": "WNLI",
    "wsc": "WSC",
    "axb": "AX-b",
    "axg": "AX-g",
}

# tfds_names that use text predictions
USES_TEXT = [
    "cb", "rte", "mnli_matched", "mnli_mismatched", "qnli", "axb", "axg"
]

# Placeholder for seq len - required by get_dataset but not used
_FAKE_LEN = {"inputs": 512, "targets": 512}


def add_global_cache_dirs():
    """Add additional task cache directories."""
    t5.data.add_global_cache_dirs(FLAGS.additional_task_cache_dirs)


def get_output_file_path():
    """Get the output file path."""
    return os.path.join(
        FLAGS.out_dir, "{}.{{extension}}".format(FILE_NAME_MAP[FLAGS.tfds_name])
    )


def load_dataset():
    """Load the dataset."""
    ds = t5.data.TaskRegistry.get_dataset(
        FLAGS.task, _FAKE_LEN, FLAGS.split, use_cached=FLAGS.cached, shuffle=False
    )
    examples = [{k: v.numpy() for k, v in ex.items()} for ex in ds]
    return examples


def read_prediction_lines():
    """Read the prediction lines from the file."""
    with tf.io.gfile.GFile(FLAGS.predictions_file) as f:
        prediction_lines = f.readlines()
    return prediction_lines


def process_predictions(predictions):
    """Process predictions based on tfds_name."""
    if FLAGS.tfds_name in USES_TEXT:
        if FLAGS.super:
            builder_configs = tfds.text.super_glue.SuperGlue.builder_configs
        else:
            builder_configs = tfds.text.glue.Glue.builder_configs
        label_classes = builder_configs[FLAGS.tfds_name].label_classes
        predictions = [label_classes[p] for p in predictions]
    elif FLAGS.tfds_name in ["boolq", "wic"]:
        predictions = [("false", "true")[p] for p in predictions]
    elif FLAGS.tfds_name == "wsc":
        predictions = [("False", "True")[p] for p in predictions]
    elif FLAGS.tfds_name == "multirc":
        rows = collections.defaultdict(lambda: collections.defaultdict(dict))
        predictions = [int(p["value"]) for p in predictions]
        for p, e in zip(predictions, examples):
            e = {k: int(e["idx/" + k]) for k in ["paragraph", "question", "answer"]}
            rows[e["paragraph"]][e["question"]][e["answer"]] = p
        with tf.io.gfile.GFile(out_file.format(extension="jsonl"), "w") as f:
            for pidx, passage in rows.items():
                qs = [
                    {"idx": i, "answers": [{"idx": j, "label": q[j]} for j in q]}
                    for i, q in passage.items()
                ]
                f.write(
                    json.dumps({"idx": pidx, "passage": {"questions": qs}}) + os.linesep
                )
        return predictions
    return predictions


def validate_predictions(predictions, examples):
    """Validate predictions against examples."""
    if len(predictions) != len(examples):
        raise ValueError(
            "Number of predictions in {} ({}) != of examples in {} split of {} "
            "({}).".format(
                FLAGS.predictions_file,
                len(predictions),
                FLAGS.split,
                FLAGS.task,
                len(examples),
            )
        )


def write_output_file(indices, predictions):
    """Write the output file."""
    if FLAGS.super:
        lines = [
            json.dumps({"idx": int(i), "label": p}) + os.linesep
            for i, p in zip(indices, predictions)
        ]
        with tf.io.gfile.GFile(out_file.format(extension="jsonl"), "w") as f:
            for line in lines:
                f.write(line)
    else:
        with tf.io.gfile.GFile(out_file.format(extension="tsv"), "w") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow(["index", "prediction"])
            tsv_writer.writerows([i, p] for i, p in zip(indices, predictions))


def main(_):
    # Add additional cache directories
    add_global_cache_dirs()

    # Get the output file path
    out_file = get_output_file_path()

    # Load the dataset
    examples = load_dataset()

    # Read the prediction lines
    prediction_lines = read_prediction_lines()

    if FLAGS.tfds_name == "record":
        predictions = [l.strip() for l in prediction_lines]
    else:
        predictions = [ast.literal_eval(l.strip()) for l in prediction_lines]

    # Process the predictions
    predictions = process_predictions(predictions)

    # Validate predictions
    validate_predictions(predictions, examples)

    # Write the output file
    if "record" in FLAGS.task:
        indices = [ex["idx/query"] for ex in examples]
    else:
        indices = [ex.get("idx", None) for ex in examples]
    write_output_file(indices, predictions)


if __name__ == "__main__":
    app.run(main)