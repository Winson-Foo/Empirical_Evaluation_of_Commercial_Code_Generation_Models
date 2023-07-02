#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def get_color_constants():
    return {
        'black': tf.constant([[0, 0, 0]], dtype=tf.int32),
        'red': tf.constant([[0, 0, 255]], dtype=tf.int32),
        'yellow': tf.constant([[0, 255, 255]], dtype=tf.int32),
        'blue': tf.constant([[255, 0, 0]], dtype=tf.int32),
        'green': tf.constant([[0, 255, 0]], dtype=tf.int32),
        'white': tf.constant([[255, 255, 255]], dtype=tf.int32)
    }


def k_means(data, target_color, bg_color1, bg_color2, alpha=1.0):
    colors = get_color_constants()
    distances = []

    for color_name in colors:
        color_distance = tf.abs(tf.subtract(data, colors[color_name]))
        if target_color == color_name:
            color_distance = tf.multiply(color_distance, alpha)
        distances.append(color_distance)

    distances = tf.concat(distances, axis=-1)
    clusters = tf.cast(tf.argmin(distances, axis=-1), tf.int32)
    mask = tf.equal(clusters, target_color)
    mask = tf.cast(mask, tf.int32)

    return mask * 255


def filter_img(img, target_color, alpha=0.9):
    colors = get_color_constants()
    # background color1
    color_1 = img[0, 0, :]
    color_1 = tf.reshape(color_1, [1, 3])
    color_1 = tf.cast(color_1, dtype=tf.int32)

    # background color2
    color_2 = img[6, 6, :]
    color_2 = tf.reshape(color_2, [1, 3])
    color_2 = tf.cast(color_2, dtype=tf.int32)

    filtered_img = k_means(img_holder, target_color, color_1, color_2, alpha)
    filtered_img = tf.expand_dims(filtered_img, axis=0)
    filtered_img = tf.expand_dims(filtered_img, axis=-1)
    filtered_img = tf.squeeze(filtered_img, name="filtered")
    return filtered_img


def compile_graph():

    with sess.graph.as_default():
        input_graph_def = sess.graph.as_graph_def()

    output_graph_def = convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names=['filtered']
    )

    last_compile_model_path = "color_extractor.pb"
    with tf.gfile.FastGFile(last_compile_model_path, mode='wb') as gf:
        gf.write(output_graph_def.SerializeToString())


if __name__ == "__main__":

    sess = tf.Session()
    img_holder = tf.placeholder(dtype=tf.int32, name="img_holder")
    color = tf.placeholder(dtype=tf.int32, name="target_color")
    filtered = filter_img(img_holder, color, alpha=0.8)

    compile_graph()