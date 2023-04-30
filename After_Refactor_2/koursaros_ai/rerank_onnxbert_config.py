MODEL_DIR = 'model_dir/'
MODEL_NAME = 'bert-base-uncased'
MODEL_TYPE = 'bert'

class ModelConfig:
    """
    Configuration for the rerank model.
    """
    def __init__(self, model_dir=MODEL_DIR, model_name=MODEL_NAME):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_type = MODEL_TYPE