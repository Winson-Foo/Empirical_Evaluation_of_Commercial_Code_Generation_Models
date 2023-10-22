from dataclasses import dataclass


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
class DeepSpeechConfig:
    limit_train_batches: int = 1
    limit_val_batches: int = 1
    max_epochs: int = 1
    batch_size: int = 10
    model_config: dict = None
    precision: int = 32
    gpus: int = 0


@dataclass
class EvalConfig:
    model_path: str = ''
    cuda: bool = False
    precision: int = 32
    lm_config: dict = None
    test_path: str = ''


@dataclass
class TranscribeConfig:
    model_path: str = ''
    cuda: bool = False
    precision: int = 32
    lm_config: dict = None
    audio_path: str = ''
