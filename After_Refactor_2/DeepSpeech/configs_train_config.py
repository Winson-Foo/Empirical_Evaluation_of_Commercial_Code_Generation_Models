from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

from omegaconf import MISSING

from deepspeech_pytorch.configs.lightning_config import TrainerConf, ModelCheckpointConf


class SpectrogramWindow(Enum):
    hamming = "hamming"
    hann = "hann"
    rectangular = "rectangular"


class RNNType(Enum):
    lstm = "lstm"
    gru = "gru"


class OptimizerType(Enum):
    sgd = "sgd"
    adam = "adam"


@dataclass
class SpectrogramConfig:
    sample_rate: int = 16000
    window_size: float = 0.02
    window_stride: float = 0.01
    window_type: SpectrogramWindow = SpectrogramWindow.hamming


@dataclass
class AugmentationConfig:
    speed_volume_perturb: bool = False
    spec_augment: bool = False
    noise_dir: Optional[str] = None
    noise_prob: float = 0.4
    noise_min: float = 0.0
    noise_max: float = 0.5


@dataclass
class DataLoaderConfig:
    train_path: str = "data/train_manifest.csv"
    val_path: str = "data/val_manifest.csv"
    batch_size: int = 64
    num_workers: int = 4
    labels_path: str = "labels.json"
    spectrogram_config: SpectrogramConfig = SpectrogramConfig()
    augmentation_config: AugmentationConfig = AugmentationConfig()
    prepare_data_per_node: bool = True


@dataclass
class RNNConfig:
    rnn_type: RNNType = RNNType.lstm
    hidden_size: int = 1024
    num_layers: int = 5


@dataclass
class BidirectionalRNNConfig(RNNConfig):
    pass


@dataclass
class UnidirectionalRNNConfig(RNNConfig):
    lookahead_context: int = 20


@dataclass
class SGDConfig:
    optimizer_type: OptimizerType = OptimizerType.sgd
    learning_rate: float = 1.5e-4
    momentum: float = 0.9
    learning_anneal: float = 0.99
    weight_decay: float = 1e-5


@dataclass
class AdamConfig:
    optimizer_type: OptimizerType = OptimizerType.adam
    learning_rate: float = 1.5e-4
    eps: float = 1e-8
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 1e-5


@dataclass
class ModelCheckpointConfig(ModelCheckpointConf):
    pass


@dataclass
class DeepSpeechTrainerConfig(TrainerConf):
    callbacks: Any = MISSING


@dataclass
class DeepSpeechConfig:
    default_configs: List[dict] = field(default_factory=lambda: [
        {"optimizer_type": "adam"},
        {"rnn_type": "bidirectional"},
        {"checkpoint": "file"}
    ])
    optimizer_config: Union[AdamConfig, SGDConfig, None] = None
    rnn_config: Union[BidirectionalRNNConfig, UnidirectionalRNNConfig, None] = None
    checkpoint_config: Union[ModelCheckpointConfig, None] = None
    trainer_config: DeepSpeechTrainerConfig = DeepSpeechTrainerConfig()
    data_loader_config: DataLoaderConfig = DataLoaderConfig()
    augmentation_config: AugmentationConfig = AugmentationConfig()
    seed: int = 123456
    load_latest_checkpoint: bool = False