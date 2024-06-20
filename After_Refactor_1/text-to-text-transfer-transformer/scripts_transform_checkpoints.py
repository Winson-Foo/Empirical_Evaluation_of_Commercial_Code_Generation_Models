import os
import re
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_list("model_dirs_or_checkpoints", [],
                  "Model directories or checkpoints to use for ensembling.")
flags.DEFINE_string("output_dir", "/tmp/",
                    "Directory to output the ensembled checkpoint to.")
flags.DEFINE_integer(
    "global_step", 0,
    "Global step to use for writing the output checkpoint file.")
flags.DEFINE_enum(
    "operation",
    "average",
    ["average", "ensemble", "autoensemble", "extract_first", "average_last_n"],
    "What to do to the input checkpoints to produce the output checkpoints")

flags.DEFINE_integer(
    "autoensemble_size", 4, "Ensemble size for 'autoensemble'")

flags.DEFINE_integer("number_of_checkpoints", 4,
                     "Number of last checkpoints for 'average_last_n'")

def average_tensors(tensors):
    return np.mean(tensors, axis=0)

def load_checkpoint(checkpoint):
    reader = tf.train.load_checkpoint(checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    var_values = {}
    var_dtypes = {}

    for (name, _) in var_list:
        tensor = reader.get_tensor(name)
        var_dtypes[name] = tensor.dtype
        var_values[name] = tensor

    return var_values, var_dtypes

def save_checkpoint(var_values, var_dtypes, output_path):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        tf_vars = [
            tf.get_variable(v, shape=var_values[v].shape, dtype=var_dtypes[v])
            for v in var_values
        ]

    placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for p, assign_op, (name, value) in zip(placeholders, assign_ops, var_values.items()):
            sess.run(assign_op, {p: value})

        saver.save(sess, output_path)

def get_latest_checkpoint(path):
    checkpoint_state = tf.train.get_checkpoint_state(path)
    return checkpoint_state.all_model_checkpoint_paths[-1]

def main(_):
    assert FLAGS.model_dirs_or_checkpoints

    if not tf.io.gfile.exists(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)

    if FLAGS.operation == "average_last_n" and len(FLAGS.model_dirs_or_checkpoints) > 1:
        raise ValueError("Need only 1 directory for average_last_n operation")

    checkpoints = []

    for path in FLAGS.model_dirs_or_checkpoints:
        if tf.io.gfile.isdir(path):
            if FLAGS.operation == "average_last_n":
                ckpt_paths = tf.io.gfile.glob(os.path.join(path, "model.ckpt*index"))
                def sort_fn(ckpt):
                    return int(re.sub(".*ckpt-", "", ckpt))

                ckpts = sorted([c.replace(".index", "") for c in ckpt_paths], key=sort_fn)
                checkpoints.extend(ckpts[-FLAGS.number_of_checkpoints:])
            else:
                checkpoints.append(get_latest_checkpoint(path))
        else:
            if FLAGS.operation == "average_last_n":
                raise ValueError("Need a directory while running average_last_n operation")
            checkpoints.append(path)

    logging.info("Using checkpoints %s", checkpoints)

    if FLAGS.operation in ["ensemble", "average", "average_last_n"]:
        if len(checkpoints) == 1:
            raise ValueError("No point in ensemble/averaging one checkpoint")
    else:
        if len(checkpoints) != 1:
            raise ValueError("Operation %s requires exactly one checkpoint" % FLAGS.operation)

    var_values = {}
    var_dtypes = {}

    for i in range(0, len(checkpoints)):
        checkpoint = checkpoints[i]
        logging.info("Loading checkpoint %s", checkpoint)
        values, dtypes = load_checkpoint(checkpoint)

        if i == 0:
            var_dtypes = dtypes

        for (name, value) in values.items():
            if i:
                assert name in var_values
                assert value.dtype == var_dtypes[name]
                var_values[name].append(value)
            else:
                var_values[name] = [value]
                if not FLAGS.global_step:
                    if name == "global_step":
                        FLAGS.global_step = value

        logging.info("Read from checkpoint %s", checkpoint)

    new_var_values = {}

    for name, tensors in var_values.items():
        tensor = tensors[0]
        if name == "global_step":
            new_val = np.int32(FLAGS.global_step)
        elif FLAGS.operation == "ensemble":
            new_val = np.stack(tensors)
        elif FLAGS.operation == "autoensemble":
            new_val = np.stack([tensor] * FLAGS.autoensemble_size)
        elif FLAGS.operation == "average" or FLAGS.operation == "average_last_n":
            new_val = average_tensors(tensors)
        elif FLAGS.operation == "extract_first":
            new_val = tensor[0]
        else:
            raise ValueError("Unknown operation: %s" % FLAGS.operation)
        
        new_var_values[name] = new_val

    save_checkpoint(new_var_values, var_dtypes, os.path.join(FLAGS.output_dir, "model.ckpt-" + str(FLAGS.global_step)))
    logging.info("Transformed checkpoints saved in %s", os.path.join(FLAGS.output_dir, "model.ckpt-" + str(FLAGS.global_step)))

if __name__ == "__main__":
    tf.disable_v2_behavior()
    app.run(main)