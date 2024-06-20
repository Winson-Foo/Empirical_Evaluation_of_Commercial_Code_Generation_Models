#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author: kerlomz <kerlomz@gmail.com>

import os
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
from config import ModelConfig

os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class GraphSession:
    def __init__(self, model_conf: ModelConfig):
        self.model_conf = model_conf
        self.logger = self.model_conf.logger
        self.size_str = self.model_conf.size_string
        self.model_name = self.model_conf.model_name
        self.graph_name = self.model_conf.model_name
        self.version = self.model_conf.model_version
        self.graph = tf.compat.v1.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph, config=self._get_session_config())
        self.graph_def = self.graph.as_graph_def()
        self.loaded = self.load_model()

    def _get_session_config(self):
        return tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True,
                per_process_gpu_memory_fraction=0.1
            )
        )

    def _import_graph(self):
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())
            _ = tf.import_graph_def(self.graph_def, name="")

    def load_model(self):
        if not self.model_conf.model_exists:
            self.destroy()
            return False
        try:
            with tf.io.gfile.GFile(self.model_conf.compile_model_path, "rb") as f:
                graph_def_file = f.read()
            self.graph_def.ParseFromString(graph_def_file)
            self._import_graph()
            self.logger.info('TensorFlow Session {} Loaded.'.format(self.model_conf.model_name))
            return True
        except NotFoundError:
            self.logger.error('The system cannot find the model specified.')
            self.destroy()
            return False

    @property
    def session(self):
        return self.sess

    def destroy(self):
        self.sess.close()
        del self.sess