import time
import tensorflow as tf


class GraphSession:

    def __init__(self, model_conf):
        self.model_conf = model_conf
        self.loaded = False
        self.graph_name = self.model_conf.graph_name
        self.version = self.model_conf.version
        self.session = None

    def load(self):
        model_path = self.model_conf.model_path
        self.session = tf.Session(graph=tf.Graph())
        with self.session.graph.as_default():
            tf.saved_model.loader.load(self.session, [tf.saved_model.tag_constants.SERVING], model_path)
        self.loaded = True

    def destroy(self):
        if self.session is not None:
            self.session.close()
        self.loaded = False