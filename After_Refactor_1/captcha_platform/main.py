import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
from model_config import ModelConfig


class GraphSession:
    def __init__(self, model_conf: ModelConfig):
        self.model_conf = model_conf
        self.logger = self.model_conf.logger
        self.graph = tf.Graph()
        self.sess = tf.Session(
            graph=self.graph,
            config=tf.ConfigProto(
                gpu_options=tf.GPUOptions(
                    allow_growth=True,
                    per_process_gpu_memory_fraction=self.model_conf.device_usage
                )
            )
        )
        self.loaded = self.load_model()

    def load_model(self):
        if not self.model_conf.model_exists:
            self.destroy()
            return False

        try:
            with tf.gfile.GFile(self.model_conf.compile_model_path, "rb") as f:
                graph_def_file = f.read()
            self.graph_def.ParseFromString(graph_def_file)
            with self.graph.as_default():
                self.sess.run(tf.global_variables_initializer())
                _ = tf.import_graph_def(self.graph_def, name="")

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