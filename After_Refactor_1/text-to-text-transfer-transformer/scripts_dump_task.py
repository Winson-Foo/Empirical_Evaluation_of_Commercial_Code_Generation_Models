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

r"""Utility to print the text or tokens in a task.

Example usage:
python -m t5.scripts.dump_task \
    --task=glue_mnli_v002 \
    --max_examples=100

"""

import importlib
import re

from absl import app
from absl import flags

import gin
import seqio

import tensorflow.compat.v1 as tf

tf.compat.v1.enable_eager_execution()

FLAGS = flags.FLAGS

flags.DEFINE_string("task", None, "A registered Task.")
flags.DEFINE_integer("max_examples", -1,
                     "maximum number of examples. -1 for no limit")
flags.DEFINE_string("format_string", "{inputs}\t{targets}",
                    "format for printing examples")
flags.DEFINE_string("split", "train",
                    "which split of the dataset, e.g. train or validation")

flags.DEFINE_bool("detokenize", False, "If True, then decode ids to strings.")
flags.DEFINE_bool("shuffle", True, "Whether to shuffle dataset or not.")
flags.DEFINE_bool("apply_postprocess_fn", False,
                  "Whether to apply the postprocess function or not.")
flags.DEFINE_bool("pretty", False, "Whether to print a pretty output.")
flags.DEFINE_multi_string(
    "delimiters", [], "Optional delimiters to highlight in terminal output when pretty is enabled."
)


@gin.configurable
def sequence_length(value=512):
    """Sequence length used when tokenizing.

    Args:
        value: an integer or dictionary

    Returns:
        a dictionary
    """
    if isinstance(value, int):
        return {"inputs": value, "targets": value}
    else:
        return value


def apply_postprocess_fn(task_or_mixture, value):
    """Apply postprocess function to the value."""
    if hasattr(task_or_mixture, "postprocess_fn"):
        return task_or_mixture.postprocess_fn(value)
    return value


def detokenize_value(value, task_or_mixture):
    """Decode the value from token IDs to strings."""
    try:
        value = task_or_mixture.output_features[k].vocabulary.decode_tf(tf.abs(value))
    except RuntimeError as err:
        value = f"Error {err} while decoding {value}"
    value = apply_postprocess_fn(task_or_mixture, value) if FLAGS.apply_postprocess_fn else value
    return value


def import_modules(modules):
    """Import specified modules."""
    for module in modules:
        importlib.import_module(module)


def example_to_string(ex, task_or_mixture):
    """Convert an example to a printable string."""
    key_value_pairs = {}
    keys = re.findall(r"{([\w+]+)}", FLAGS.format_string)

    for key in keys:
        if key not in ex:
            key_value_pairs[key] = ""
            continue

        value = ex[key]
        if FLAGS.detokenize:
            value = detokenize_value(value, task_or_mixture)

        if tf.rank(value) == 0:
            value = [value]

        if tf.is_numeric_tensor(value):
            value = tf.strings.format("{}", tf.squeeze(value), summarize=-1)
        else:
            value = tf.strings.join(value, separator="\n\n")
        
        key_value_pairs[key] = pretty(value.numpy().decode("utf-8"))

    return FLAGS.format_string.format(**key_value_pairs)


def pretty(value):
    """Optional pretty printing helper for detokenized inputs.

    Makes any text delimiter regex specified in `--delimiters` bold in textual
    output.

    Args:
        value: string representing the detokenized output

    Returns:
        a string with appropriate styling applied
    """
    if not FLAGS.pretty or not FLAGS.detokenize:
        return value

    combined_matcher = re.compile(f"({'|'.join(FLAGS.delimiters)})")
    return combined_matcher.sub(u"\u001b[1m\\1\u001b[0m", value)


def main(_):
    flags.mark_flags_as_required(["task"])

    if FLAGS.module_import:
        import_modules(FLAGS.module_import)

    total_examples = 0
    if FLAGS.task is not None:
        task_or_mixture = seqio.TaskRegistry.get(FLAGS.task)
    elif FLAGS.mixture is not None:
        task_or_mixture = seqio.MixtureRegistry.get(FLAGS.mixture)

    ds = task_or_mixture.get_dataset(
        sequence_length=sequence_length(),
        split=FLAGS.split,
        use_cached=False,
        shuffle=FLAGS.shuffle)

    for ex in ds:
        print(example_to_string(ex, task_or_mixture))
        total_examples += 1
        if total_examples == FLAGS.max_examples:
            break


if __name__ == "__main__":
    app.run(main)