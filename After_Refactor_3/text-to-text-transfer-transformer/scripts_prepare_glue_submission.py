from typing import List, Dict
from pathlib import Path
import ast
import collections
import csv
import json

from absl import app
from absl import flags
import t5.data
import tensorflow as tf
import tensorflow_datasets as tfds

FLAGS = flags.FLAGS

flags.DEFINE_string("predictions_file", None, "Path to model predictions.")
flags.DEFINE_string("task", None, "T5 task name for this benchmark.")
flags.DEFINE_string("tfds_name", None, "Short name of tfds (e.g. 'cb').")
flags.DEFINE_string("out_dir", None, "Path to write output file.")
flags.DEFINE_string("split", "test", "Split, should typically be test.")
flags.DEFINE_boolean("super", False, "Whether to make SuperGLUE-style file.")
flags.DEFINE_boolean("cached", True, "Whether to use cached dataset.")
flags.DEFINE_list("additional_task_cache_dirs", [], "Dirs with cached tasks.")

def prepare_for_submission():
    t5.data.add_global_cache_dirs(FLAGS.additional_task_cache_dirs)
    out_file = Path(FLAGS.out_dir) / f"{get_mapped_task_name()}.{{extension}}"

    examples = load_dataset_examples()
    predictions = load_predictions()

    if len(predictions) != len(examples):
        raise ValueError(
            f"Number of predictions in {FLAGS.predictions_file} "
            f"({len(predictions)}) != of examples in {FLAGS.split} split of "
            f"{FLAGS.task} ({len(examples)})."
        )

    prepare_predictions(out_file, predictions, examples)

def get_mapped_task_name() -> str:
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
    return FILE_NAME_MAP[FLAGS.tfds_name]

def load_dataset_examples() -> List[Dict[str, tf.Tensor]]:
    dataset = t5.data.TaskRegistry.get_dataset(
        FLAGS.task, _FAKE_LEN, FLAGS.split, use_cached=FLAGS.cached, shuffle=False
    )
    return [{k: v.numpy() for k, v in ex.items()} for ex in dataset]

def load_predictions() -> List:
    with open(FLAGS.predictions_file) as f:
        prediction_lines = f.readlines()

    if FLAGS.tfds_name == "record":
        predictions = [l.strip() for l in prediction_lines]
    else:
        predictions = [ast.literal_eval(l.strip()) for l in prediction_lines]

    if FLAGS.tfds_name in USES_TEXT:
        label_classes = get_label_classes()
        predictions = [label_classes[p] for p in predictions]
    elif FLAGS.tfds_name in ["boolq", "wic"]:
        predictions = [("false", "true")[p] for p in predictions]
    elif FLAGS.tfds_name == "wsc":
        predictions = [("False", "True")[p] for p in predictions]
    elif FLAGS.tfds_name == "multirc":
        predictions = prepare_multirc_predictions(predictions)

    return predictions

def get_label_classes() -> List[str]:
    if FLAGS.super:
        builder_configs = tfds.text.super_glue.SuperGlue.builder_configs
    else:
        builder_configs = tfds.text.glue.Glue.builder_configs
    return builder_configs[FLAGS.tfds_name].label_classes

def prepare_multirc_predictions(predictions: List) -> List[int]:
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    predictions = [int(p["value"]) for p in predictions]

    for p, e in zip(predictions, examples):
        e = {k: int(e["idx/" + k]) for k in ["paragraph", "question", "answer"]}
        rows[e["paragraph"]][e["question"]][e["answer"]] = p

    output_file = out_file.format(extension="jsonl")
    with open(output_file, "w") as f:
        for pidx, passage in rows.items():
            qs = [
                {"idx": i, "answers": [{"idx": j, "label": q[j]} for j in q]}
                for i, q in passage.items()
            ]
            f.write(json.dumps({"idx": pidx, "passage": {"questions": qs}}) + os.linesep)

def prepare_predictions(out_file: Path, predictions: List, examples: List[Dict[str, tf.Tensor]]):
    indices = [ex.get("idx", None) for ex in examples]

    if FLAGS.super:
        lines = [
            json.dumps({"idx": int(i), "label": p}) + os.linesep
            for i, p in zip(indices, predictions)
        ]
        with open(out_file.format(extension="jsonl"), "w") as f:
            for line in lines:
                f.write(line)
    else:
        with open(out_file.format(extension="tsv"), "w") as out_file:
            tsv_writer = csv.writer(out_file, delimiter="\t")
            tsv_writer.writerow(["index", "prediction"])
            tsv_writer.writerows([i, p] for i, p in zip(indices, predictions))

def main(_):
    prepare_for_submission()

if __name__ == "__main__":
    app.run(main)