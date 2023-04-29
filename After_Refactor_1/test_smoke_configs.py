from dataclasses import dataclass
from deepspeech_pytorch.enums import DecoderType

@dataclass
class DatasetConfig:
    target_dir: str = ''
    manifest_dir: str = ''
    min_duration: float = 0
    max_duration: float = 15
    val_fraction: float = 0.1
    sample_rate: int = 16000
    num_workers: int = 4
    
@dataclass
class EvalConfig:
    model: ModelConfig
    lm: LMConfig
    test_path: str

@dataclass
class TranscribeConfig:
    model: ModelConfig
    lm: LMConfig
    audio_path: str

@dataclass
class DeepSpeechConfig:
    trainer: DeepSpeechTrainerConf
    data: DataConfig
    optim: AdamConfig
    model: BiDirectionalConfig
    checkpoint: ModelCheckpointConf

@dataclass
class ModelConfig:
    cuda: bool
    model_path: str
    precision: int

@dataclass
class LMConfig:
    alpha: float = 0.75
    beta: float = 1.85
    cutoff_top_n: int = 40
    cutoff_prob: float = 1.0
    beam_size: int = 5000
    lm_path: str = ''
    decoder_type: DecoderType = DecoderType.beam
    
@dataclass
class DeepSpeechTrainerConf:
    max_epochs: int
    precision: int
    gpus: int
    enable_checkpointing: bool
    limit_train_batches: int
    limit_val_batches: int
    
@dataclass
class DataConfig:
    train_path: str
    val_path: str
    batch_size: int

@dataclass
class AdamConfig:
    learning_rate: float = 3e-4
    betas: tuple = (0.8, 0.9)
    eps: float = 1e-9
    weight_decay: float = 1e-6
    
@dataclass
class BiDirectionalConfig:
    rnn_type: str = 'LSTM'
    hidden_size: int = 768
    hidden_layers: int = 5
    dropout: float = 0.1
    bidirectional: bool = True