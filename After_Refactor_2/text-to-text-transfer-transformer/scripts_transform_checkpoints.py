import os
import re
from absl import app, flags, logging
import numpy as np
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_list("model_dirs_or_checkpoints", [],
                  "Model directories or checkpoints to use for ensembling.")
flags.DEFINE_string("output_dir", "/tmp/",
                    "Directory to output the ensembled checkpoint to.")
flags.DEFINE_integer("global_step", 0,
                     "Global step to use for writing the output checkpoint file.")
flags.DEFINE_enum("operation", "average",
                  ["average", "ensemble", "autoensemble", "extract_first", "average_last_n"],
                  "what to do to the input checkpoints to produce the output checkpoints")

flags.DEFINE_integer("autoensemble_size", 4, "ensemble size for 'autoensemble'")
flags.DEFINE_integer("number_of_checkpoints", 4,
                     "number of last checkpoints for 'average_last_n'")


def average_tensors(tensors):
    result = tensors[0]
    for t in tensors[1:]:
        result += t
    return result / len(tensors)


def extract_last_n_checkpoints(model_dir, n):
    ckpt_paths = tf.io.gfile.glob(os.path.join(model_dir, "model.ckpt*index"))
    def sort_fn(ckpt):
        return int(re.sub(".*ckpt-", "", ckpt))

    ckpts = sorted([c.replace(".index", "") for c in ckpt_paths], key=sort_fn)
    return ckpts[-n:]


def get_latest_checkpoint(model_dir):
    checkpoint_state = tf.train.get_checkpoint_state(model_dir)
    return checkpoint_state.all_model_checkpoint_paths[-1]


def load_checkpoint(checkpoint):
    reader = tf.train.load_checkpoint(checkpoint)
    var_list = tf.train.list_variables(checkpoint)
    var_values = {}
    for (name, _) in var_list:
        tensor = reader.get_tensor(name)
        var_values[name] = tensor
    return var_values


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


def transform_checkpoint(model_dirs_or_checkpoints, output_dir, global_step, operation):
    assert model_dirs_or_checkpoints

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    if operation == "average_last_n" and len(model_dirs_or_checkpoints) > 1:
        raise ValueError("Need only 1 directory for average_last_n operation")

    checkpoints = []

    for path in model_dirs_or_checkpoints:
        if tf.io.gfile.isdir(path):
            if operation == "average_last_n":
              checkpoints.extend(extract_last_n_checkpoints(path, FLAGS.number_of_checkpoints))
            else:
              checkpoints.append(get_latest_checkpoint(path))
        else:
            if operation == "average_last_n":
                raise ValueError("need a directory while running average_last_n operation")
            checkpoints.append(path)

    logging.info("Using checkpoints %s", checkpoints)

    if operation in ["ensemble", "average", "average_last_n"]:
        if len(checkpoints) == 1:
            raise ValueError("no point in ensemble/averaging one checkpoint")
    else:
        if len(checkpoints) != 1:
            raise ValueError("operation %s requires exactly one checkpoint" % operation)

    var_values = {}
    var_dtypes = {}

    for i in range(len(checkpoints)):
        checkpoint = checkpoints[i]
        logging.info("loading checkpoint %s", checkpoint)
        var_values.update(load_checkpoint(checkpoint))
        logging.info("Read from checkpoint %s", checkpoint)

    new_var_values = {}

    for name, tensors in var_values.items():
        tensor = tensors[0]
        if name == "global_step":
            new_val = np.int32(global_step)
        elif operation == "ensemble":
            new_val = np.stack(tensors)
        elif operation == "autoensemble":
            new_val = np.stack([tensor] * FLAGS.autoensemble_size)
        elif operation == "average" or operation == "average_last_n":
            new_val = average_tensors(tensors)
        elif operation == "extract_first":
            new_val = tensor[0]
        else:
            raise ValueError("unknown operation=%s" % operation)
        new_var_values[name] = new_val

    save_checkpoint(new_var_values, var_dtypes, os.path.join(output_dir, "model.ckpt-" + str(global_step)))

    logging.info("Transformed checkpoints saved in %s", output_path)


def main(_):
    tf.disable_v2_behavior()
    transform_checkpoint(FLAGS.model_dirs_or_checkpoints, FLAGS.output_dir, FLAGS.global_step, FLAGS.operation)


if __name__ == "__main__":
    app.run(main)