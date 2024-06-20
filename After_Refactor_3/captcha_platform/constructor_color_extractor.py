#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

COLOR_MAP = {
    "black": tf.constant([[0, 0, 0]], dtype=tf.int32),
    "red": tf.constant([[0, 0, 255]], dtype=tf.int32),
    "yellow": tf.constant([[0, 255, 255]], dtype=tf.int32),
    "blue": tf.constant([[255, 0, 0]], dtype=tf.int32),
    "green": tf.constant([[0, 255, 0]], dtype=tf.int32),
    "white": tf.constant([[255, 255, 255]], dtype=tf.int32),
}


def get_distance(data, point):
    sum_squares = tf.cast(tf.reduce_sum(tf.abs(tf.subtract(data, point)), axis=2, keep_dims=True), tf.float32)
    return sum_squares


def get_distance_for_color(data, target_color, alpha):
    color_distance = get_distance(data, COLOR_MAP[target_color])
    if target_color == "red":
        color_distance = tf.multiply(color_distance, tf.constant(alpha, dtype=tf.float32))
    return color_distance


def k_means(data, target_color, bg_color1, bg_color2, alpha=1.0):
    black_distance = get_distance_for_color(data, "black", alpha)
    red_distance = get_distance_for_color(data, "red", alpha)
    blue_distance = get_distance_for_color(data, "blue", alpha)
    yellow_distance = get_distance_for_color(data, "yellow", alpha)
    green_distance = get_distance_for_color(data, "green", alpha)
    white_distance = get_distance_for_color(data, "white", alpha)

    c_1_distance = get_distance(data, bg_color1)
    c_2_distance = get_distance(data, bg_color2)

    distances = tf.concat([
        black_distance,
        red_distance,
        blue_distance,
        yellow_distance,
        green_distance,
        c_1_distance,
        c_2_distance,
        white_distance
    ], axis=-1)

    clusters = tf.cast(tf.argmin(distances, axis=-1), tf.int32)

    mask = tf.equal(clusters, tf.constant(target_color, dtype=tf.int32))
    mask = tf.cast(mask, tf.int32)

    return mask * 255


def filter_img(img, target_color, alpha=0.9):
    # background color1
    bg_color1 = img[0, 0, :]
    bg_color1 = tf.reshape(bg_color1, [1, 3])
    bg_color1 = tf.cast(bg_color1, dtype=tf.int32)

    # background color2
    bg_color2 = img[6, 6, :]
    bg_color2 = tf.reshape(bg_color2, [1, 3])
    bg_color2 = tf.cast(bg_color2, dtype=tf.int32)

    filtered_img = k_means(img_holder, target_color, bg_color1, bg_color2, alpha)
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
        # gf.write(output_graph_def.SerializeToString())
        print(output_graph_def.SerializeToString())


if __name__ == "__main__":
    sess = tf.Session()
    img_holder = tf.placeholder(dtype=tf.int32, name="img_holder")
    target_color = tf.placeholder(dtype=tf.int32, name="target_color")
    filtered = filter_img(img_holder, target_color, alpha=0.8)
    compile_graph()