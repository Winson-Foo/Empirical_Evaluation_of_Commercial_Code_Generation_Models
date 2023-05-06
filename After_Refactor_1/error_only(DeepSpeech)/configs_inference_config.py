from dataclasses import dataclass
from typing import Union
from deepspeech_pytorch.enums import DecoderType

@dataclass
class LMConfig:
    decoder_type: DecoderType = DecoderType.greedy
    lm_path: Union[str, None] = None
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
    model_path: Union[str, None] = None

@dataclass
class InferenceConfig:
    lm: LMConfig = LMConfig()
    model: ModelConfig = ModelConfig()

@dataclass
class TranscribeConfig(InferenceConfig):
    audio_path: Union[str, None] = None
    offsets: bool = False
    chunk_size_seconds: float = -1

@dataclass
class EvalConfig(InferenceConfig):
    test_path: Union[str, None] = None
    verbose: bool = True
    save_output: Union[str, None] = None
    batch_size: int = 20
    num_workers: int = 4

@dataclass
class ServerConfig(InferenceConfig):
    host: str = '0.0.0.0'
    port: int = 8888

# Error handling
# Checking if the given path exists for the model and audio
def validate_config(config):
    if isinstance(config, TranscribeConfig):
        if not config.audio_path:
            raise ValueError("audio_path is not provided.")
        if not config.model.model_path:
            raise ValueError("model_path is not provided.")
    elif isinstance(config, EvalConfig):
        if not config.test_path:
            raise ValueError("test_path is not provided.")
        if not config.model.model_path:
            raise ValueError("model_path is not provided.")

transcribe_config = TranscribeConfig()
eval_config = EvalConfig()

validate_config(transcribe_config)
validate_config(eval_config)