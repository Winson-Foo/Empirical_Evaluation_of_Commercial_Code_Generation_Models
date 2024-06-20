import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class GraphSession:
    def __init__(self, model_conf, session, graph_name, version):
        self.model_conf = model_conf
        self.session = session
        self.graph_name = graph_name
        self.version = version
        self.loaded = True

    def destroy(self):
        self.session.close()
        self.loaded = False