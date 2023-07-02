import os


class ModelConfig:
    def __init__(self, logger):
        self.logger = logger
        self.size_string = ""
        self.model_name = ""
        self.model_version = ""
        self.compile_model_path = ""
        self.model_exists = False
        self.device_usage = 0.1


from model_config import ModelConfig
from graph_session import GraphSession

config = ModelConfig(logger)
graph_session = GraphSession(config)
session = graph_session.session
# Use the session object as needed.