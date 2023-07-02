import os
import re
import numpy as np
import tensorflow.compat.v1 as tf

from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_list("model_dirs_or_checkpoints", [], "Model directories or checkpoints to use for ensembling.")
flags.DEFINE_string("output_dir", "/tmp/", "Directory to output the ensembled checkpoint to.")
flags.DEFINE_integer("global_step", 0, "Global step to use for writing the output checkpoint file.")
flags.DEFINE_enum("operation", "average", ["average", "ensemble", "autoensemble", "extract_first", "average_last_n"], "What to do to the input checkpoints to produce the output checkpoints")
flags.DEFINE_integer("autoensemble_size", 4, "Ensemble size for 'autoensemble'")
flags.DEFINE_integer("number_of_checkpoints", 4, "Number of last checkpoints for 'average_last_n'")

def average_tensors(tensors):
    """Utility function to average tensors."""
    result = tensors[0]
    for t in tensors[1:]:
        result += t
    return result / len(tensors)

def get_latest_checkpoint(directory):
    """Get the latest checkpoint file from a model directory."""
    checkpoint_state = tf.train.get_checkpoint_state(directory)
    return checkpoint_state.all_model_checkpoint_paths[-1]

def get_checkpoint_paths(directory, number_of_checkpoints):
    """Get the paths of the last N checkpoints in a model directory."""
    ckpt_paths = tf.io.gfile.glob(os.path.join(directory, "model.ckpt*index"))
    def sort_fn(ckpt):
        return int(re.sub(".*ckpt-", "", ckpt))
    sorted_ckpts = sorted([c.replace(".index", "") for c in ckpt_paths], key=sort_fn)
    return sorted_ckpts[-number_of_checkpoints:]

def load_checkpoint(checkpoint):
    """Load checkpoint and retrieve variable values and dtypes."""
    reader = tf.train.load_checkpoint(checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    var_values = {}
    var_dtypes = {}
    for (name, _) in var_list:
        tensor = reader.get_tensor(name)
        var_values[name] = tensor
        var_dtypes[name] = tensor.dtype
    return var_values, var_dtypes

def save_checkpoint(var_values, var_dtypes, output_path, global_step):
    """Save checkpoint using variable values and dtypes."""
    tf_vars = []
    placeholders = []
    assign_ops = []
    for name, value in var_values.items():
        dtype = var_dtypes[name]
        shape = value.shape
        tf_var = tf.get_variable(name, shape=shape, dtype=dtype)
        tf_vars.append(tf_var)
        placeholders.append(tf.placeholder(dtype, shape=shape))
        assign_ops.append(tf.assign(tf_var, placeholders[-1]))

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for placeholder, assign_op, (name, value) in zip(placeholders, assign_ops, var_values.items()):
            sess.run(assign_op, {placeholder: value})

        saver.save(sess, output_path, global_step=global_step)

def ensemble_checkpoints(checkpoints):
    """Ensemble checkpoints by stacking tensors."""
    var_values, var_dtypes = load_checkpoint(checkpoints[0])

    for i in range(1, len(checkpoints)):
        checkpoint = checkpoints[i]
        var_values_i, var_dtypes_i = load_checkpoint(checkpoint)
        for name in var_values.keys():
            tensor = var_values_i[name]
            var_values[name] = np.concatenate([var_values[name], tensor], axis=0)

    return var_values, var_dtypes

def average_checkpoints(checkpoints):
    """Average checkpoints by taking the arithmetic mean of tensors."""
    var_values, var_dtypes = load_checkpoint(checkpoints[0])

    for i in range(1, len(checkpoints)):
        checkpoint = checkpoints[i]
        var_values_i, _ = load_checkpoint(checkpoint)
        for name in var_values.keys():
            tensor_i = var_values_i[name]
            var_values[name] += tensor_i

    for name in var_values.keys():
        var_values[name] /= len(checkpoints)

    return var_values, var_dtypes

def autoensemble_checkpoints(checkpoints, ensemble_size):
    """Create an ensemble of identical models from one checkpoint."""
    var_values, var_dtypes = load_checkpoint(checkpoints[0])
    for name in var_values.keys():
        tensor = var_values[name]
        var_values[name] = np.stack([tensor] * ensemble_size)

    return var_values, var_dtypes

def extract_first_checkpoint(checkpoints):
    """Extract the first element of an ensemble checkpoint."""
    var_values, var_dtypes = load_checkpoint(checkpoints[0])
    for name in var_values.keys():
        tensor = var_values[name]
        var_values[name] = tensor[0]

    return var_values, var_dtypes

def average_last_n_checkpoints(directory, number_of_checkpoints):
    """Average the last N checkpoints in the given directory."""
    checkpoints = get_checkpoint_paths(directory, number_of_checkpoints)
    var_values, var_dtypes = average_checkpoints(checkpoints)
    return var_values, var_dtypes

def transform_checkpoints():
    """Main function to transform checkpoints based on the specified operation."""
    assert FLAGS.model_dirs_or_checkpoints

    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)

    checkpoints = []

    for path in FLAGS.model_dirs_or_checkpoints:
        if tf.io.gfile.isdir(path):
            if FLAGS.operation == "average_last_n":
                checkpoints.extend(get_checkpoint_paths(path, FLAGS.number_of_checkpoints))
            else:
                checkpoints.append(get_latest_checkpoint(path))
        else:
            if FLAGS.operation == "average_last_n":
                raise ValueError("Need a directory while running %s operation" % FLAGS.operation)
            checkpoints.append(path)

    logging.info("Using checkpoints %s", checkpoints)

    if FLAGS.operation in ["ensemble", "average", "average_last_n"]:
        if len(checkpoints) == 1:
            raise ValueError("No point in ensembling/averaging one checkpoint")
    else:
        if len(checkpoints) != 1:
            raise ValueError("Operation %s requires exactly one checkpoint" % FLAGS.operation)

    operation_mapping = {
        "ensemble": ensemble_checkpoints,
        "average": average_checkpoints,
        "autoensemble": lambda c: autoensemble_checkpoints(c, FLAGS.autoensemble_size),
        "extract_first": extract_first_checkpoint,
        "average_last_n": lambda c: average_last_n_checkpoints(c[0], FLAGS.number_of_checkpoints)
    }

    var_values, var_dtypes = operation_mapping[FLAGS.operation](checkpoints)

    output_file = "model.ckpt-" + str(FLAGS.global_step)
    output_path = os.path.join(FLAGS.output_dir, output_file)

    save_checkpoint(var_values, var_dtypes, output_path, FLAGS.global_step)

    logging.info("Transformed checkpoints saved in %s", output_path)

def main(_):
    tf.disable_v2_behavior()
    transform_checkpoints()

if __name__ == "__main__":
    app.run(main)