from dataclasses import dataclass
from deepspeech_pytorch.enums import DecoderType


@dataclass
class LanguageModelConfig:
    decoder_type: DecoderType = DecoderType.greedy
    path: str = ''
    top_beams: int = 1
    weight: float = 0.0
    word_bonus: float = 0.0
    cutoff_n: int = 40
    cutoff_prob: float = 1.0
    beam_width: int = 10
    workers: int = 4


@dataclass
class ModelInferenceConfig:
    precision_bits: int = 32
    use_gpu: bool = True
    path: str = ''


@dataclass
class AudioTranscriptionConfig:
    language_model: LanguageModelConfig = LanguageModelConfig()
    model: ModelInferenceConfig = ModelInferenceConfig()
    audio_file: str = ''
    return_offsets: bool = False
    chunk_size_in_seconds: float = -1.0


@dataclass
class ModelEvaluationConfig:
    language_model: LanguageModelConfig = LanguageModelConfig()
    model: ModelInferenceConfig = ModelInferenceConfig()
    test_data: str = ''
    verbose_output: bool = True
    output_file: str = ''
    batch_size: int = 20
    num_workers: int = 4


@dataclass
class ModelServerConfig:
    language_model: LanguageModelConfig = LanguageModelConfig()
    model: ModelInferenceConfig = ModelInferenceConfig()
    host: str = '0.0.0.0'
    port: int = 8888
