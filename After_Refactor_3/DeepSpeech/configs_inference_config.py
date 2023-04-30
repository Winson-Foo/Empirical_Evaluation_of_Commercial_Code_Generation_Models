from dataclasses import dataclass
from deepspeech_pytorch.enums import DecoderType


@dataclass
class LMConfig:
    decoder_type: DecoderType = DecoderType.greedy
    lm_path: str = ''
    top_paths: int = 1
    alpha: float = 0.0
    beta: float = 0.0
    cutoff_top_n: int = 40
    cutoff_prob: float = 1.0
    beam_width: int = 10
    lm_workers: int = 4


@dataclass
class ModelConfig:
    precision: int = 32
    cuda: bool = True
    model_path: str = ''


@dataclass
class InferenceConfig:
    lm: LMConfig
    model: ModelConfig


@dataclass
class TranscribeConfig(InferenceConfig):
    audio_path: str = ''
    offsets: bool = False
    chunk_size_seconds: float = -1


@dataclass
class EvalConfig(InferenceConfig):
    test_path: str = ''
    verbose: bool = True
    save_output: str = ''
    batch_size: int = 20
    num_workers: int = 4


@dataclass
class ServerConfig(InferenceConfig):
    host: str = '0.0.0.0'
    port: int = 8888


class LM:
    def __init__(self, config: LMConfig):
        self.config = config
        
    def get_results(self, inputs):
        pass  # TODO: implement


class Model:
    def __init__(self, config: ModelConfig):
        self.config = config
    
    def predict(self, inputs):
        pass  # TODO: implement


class Transcriber:
    def __init__(self, config: TranscribeConfig):
        self.config = config
        self.lm = LM(config.lm)
        self.model = Model(config.model)
    
    def transcribe(self):
        pass  # TODO: implement


class Evaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.lm = LM(config.lm)
        self.model = Model(config.model)
    
    def evaluate(self):
        pass  # TODO: implement


class Server:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.lm = LM(config.lm)
        self.model = Model(config.model)
    
    def start(self):
        pass  # TODO: implement