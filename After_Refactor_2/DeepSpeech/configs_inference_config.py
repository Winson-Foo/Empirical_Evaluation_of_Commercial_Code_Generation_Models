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
    path: str = ''


@dataclass
class AudioConfig:
    path: str = ''
    chunk_size_seconds: float = -1


@dataclass
class DecoderConfig:
    lm: LMConfig = LMConfig()
    alpha: float = 0.0
    beta: float = 0.0
    beam_width: int = 10


@dataclass
class EvalConfig:
    model: ModelConfig = ModelConfig()
    decoder: DecoderConfig = DecoderConfig()
    test_path: str = ''
    verbose: bool = True
    save_output: str = ''
    batch_size: int = 20
    num_workers: int = 4


@dataclass
class ServerConfig:
    model: ModelConfig = ModelConfig()
    decoder: DecoderConfig = DecoderConfig()
    audio: AudioConfig = AudioConfig()
    host: str = '0.0.0.0'
    port: int = 8888