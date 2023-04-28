import functools

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional


class SpectrogramWindow(Enum):
    hamming = 'hamming'
    hann = 'hann'
    blackman = 'blackman'


class RNNType(Enum):
    lstm = 'lstm'
    gru = 'gru'


@dataclass
class SpectConfig:
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    window: SpectrogramWindow = SpectrogramWindow.hamming


@dataclass
class AugmentationConfig:
    speed_volume_perturb: bool = False
    spec_augment: bool = False
    noise_dir: str = ''
    noise_prob: float = 0.4
    noise_min: float = 0.0
    noise_max: float = 0.5


@dataclass
class DataConfig:
    train_path: str = 'data/train_manifest.csv'
    val_path: str = 'data/val_manifest.csv'
    batch_size: int = 64
    num_workers: int = 4
    labels_path: str = 'labels.json'
    spect: SpectConfig = field(default_factory=SpectConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    prepare_data_per_node: bool = True


@dataclass
class RNNConfig:
    rnn_type: RNNType = RNNType.lstm
    hidden_size: int = 1024
    hidden_layers: int = 5


@dataclass
class BiDirectionalConfig(RNNConfig):
    pass


@dataclass
class UniDirectionalConfig(RNNConfig):
    lookahead_context: int = 20


@dataclass
class OptimConfig:
    learning_rate: float = 1.5e-4
    learning_anneal: float = 0.99
    weight_decay: float = 1e-5


@dataclass
class SGDConfig(OptimConfig):
    momentum: float = 0.9


@dataclass
class AdamConfig(OptimConfig):
    eps: float = 1e-8
    betas: tuple = (0.9, 0.999)


@dataclass
class TrainerConf:
    max_epochs: int = 50
    gradient_clip_val: float = 5.0


@dataclass
class ModelCheckpointConf:
    dirpath: str = 'saved_models'
    monitor: str = 'val_loss'
    mode: str = 'min'


@dataclass
class DeepSpeechTrainerConf(TrainerConf):
    callbacks: Any = None


@dataclass
class DeepSpeechConfig:
    defaults: List[dict] = field(default_factory=lambda: [{"optim": "adam"}, {"model": "bidirectional"}, {"checkpoint": "file"}])
    optim: Optional[Union[SGDConfig, AdamConfig]] = None
    model: Optional[Union[BiDirectionalConfig, UniDirectionalConfig]] = None
    checkpoint: Optional[ModelCheckpointConf] = None
    trainer: Optional[DeepSpeechTrainerConf] = None
    data: Optional[DataConfig] = None
    augmentation: Optional[AugmentationConfig] = None
    seed: int = 123456
    load_auto_checkpoint: bool = False
