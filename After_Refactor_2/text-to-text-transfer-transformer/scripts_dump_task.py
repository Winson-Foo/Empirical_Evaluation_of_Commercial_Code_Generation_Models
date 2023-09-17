import importlib
import re

from absl import app
from absl import flags

from mesh_tensorflow.transformer import utils
import gin
import seqio

import tensorflow.compat.v1 as tf

tf.compat.v1.enable_eager_execution()

# Define flags
FLAGS = flags.FLAGS

flags.DEFINE_string("task", None, "A registered Task.")
flags.DEFINE_string("mixture", None, "A registered Mixture.")
flags.DEFINE_integer("max_examples", -1, "maximum number of examples. -1 for no limit")
flags.DEFINE_string("format_string", "{inputs}\t{targets}", "format for printing examples")
flags.DEFINE_multi_string("module_import", [], "Modules to import.")
flags.DEFINE_string("split", "train", "which split of the dataset, e.g. train or validation")
flags.DEFINE_bool("detokenize", False, "If True, then decode ids to strings.")
flags.DEFINE_bool("shuffle", True, "Whether to shuffle dataset or not.")
flags.DEFINE_bool("apply_postprocess_fn", False, "Whether to apply the postprocess function or not.")
flags.DEFINE_bool("pretty", False, "Whether to print a pretty output.")
flags.DEFINE_multi_string("delimiters", [], "Optional delimiters to highlight in terminal output when pretty is enabled.")

# Load gin parameters if they've been defined
def load_gin_params():
    try:
        for gin_file_path in FLAGS.gin_location_prefix:
            gin.add_config_file_search_path(gin_file_path)
        gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    except AttributeError:
        # Use default settings
        gin.parse_config_files_and_bindings(None, None)

# Import required modules
def import_modules(modules):
    for module in modules:
        importlib.import_module(module)

# Convert example to string
def example_to_string(example):
    keys = re.findall(r"{([\w+]+)}", FLAGS.format_string)
    key_to_string = {}
    
    for key in keys:
        if key not in example:
            key_to_string[key] = ""
            continue
            
        value = example[key]
        
        if FLAGS.detokenize:
            try:
                value = task_or_mixture.output_features[key].vocabulary.decode_tf(tf.abs(value))
            except RuntimeError as err:
                value = f"Error {err} while decoding {value}"
            
            if FLAGS.apply_postprocess_fn and key == "targets" and hasattr(task_or_mixture, "postprocess_fn"):
                value = task_or_mixture.postprocess_fn(value)
        
        if tf.rank(value) == 0:
            value = [value]
        
        if tf.is_numeric_tensor(value):
            value = tf.strings.format("{}", tf.squeeze(value), summarize=-1)
        else:
            value = tf.strings.join(value, separator="\n\n")
        
        key_to_string[key] = pretty(value.numpy().decode("utf-8"))
    
    return FLAGS.format_string.format(**key_to_string)

# Print examples
def print_examples(ds):
    total_examples = 0
    
    for example in ds:
        print(example_to_string(example))
        total_examples += 1
        
        if total_examples == FLAGS.max_examples:
            break

# Main function
def main(_):
    flags.mark_flags_as_required(["task"])
    
    if FLAGS.module_import:
        import_modules(FLAGS.module_import)
    
    load_gin_params()
    
    if FLAGS.task:
        task_or_mixture = seqio.TaskRegistry.get(FLAGS.task)
    elif FLAGS.mixture:
        task_or_mixture = seqio.MixtureRegistry.get(FLAGS.mixture)
    
    ds = task_or_mixture.get_dataset(
        sequence_length=sequence_length(),
        split=FLAGS.split,
        use_cached=False,
        shuffle=FLAGS.shuffle
    )
    
    print_examples(ds)

if __name__ == "__main__":
    app.run(main)