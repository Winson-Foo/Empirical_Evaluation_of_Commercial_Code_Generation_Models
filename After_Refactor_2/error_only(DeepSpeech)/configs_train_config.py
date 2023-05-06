from dataclasses import dataclass, field
from typing import Any, List
from omegaconf import MISSING
from deepspeech_pytorch.configs.lightning_config import TrainerConf, ModelCheckpointConf
from deepspeech_pytorch.enums import SpectrogramWindow, RNNType

defaults = [
    {"optim": "adam"},
    {"model": "bidirectional"},
    {"checkpoint": "file"}
]

@dataclass
class SpectConfig:
    sample_rate: int = 16000   
    window_size: float = .02   
    window_stride: float = .01  
    window: SpectrogramWindow = SpectrogramWindow.hamming 

@dataclass
class AugmentationConfig:
    speed_volume_perturb: bool = False  
    spec_augment: bool = False  
    noise_dir: str =''  
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
    spect: SpectConfig = SpectConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    prepare_data_per_node: bool = True

@dataclass
class BiDirectionalConfig:
    rnn_type: RNNType = RNNType.lstm  
    hidden_size: int = 1024  
    hidden_layers: int = 5  

@dataclass
class UniDirectionalConfig(BiDirectionalConfig):
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
class DeepSpeechTrainerConf(TrainerConf):
    callbacks: Any = MISSING

@dataclass
class DeepSpeechConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    optim: Any = MISSING
    model: Any = MISSING
    checkpoint: ModelCheckpointConf = MISSING
    trainer: DeepSpeechTrainerConf = DeepSpeechTrainerConf()
    data: DataConfig = DataConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    seed: int = 123456  
    load_auto_checkpoint: bool = False  

def validate_type(instance, type_, field_name, class_name):
    if not isinstance(instance, type_):
        raise TypeError(f'Error: {class_name}.{field_name} should be of type {type_.__name__}, but got {type(instance).__name__} instead')

def validate_enum(instance, enum, field_name, class_name):
    if instance not in enum:
        raise ValueError(f'Error: {class_name}.{field_name} should be one of {", ".join(enum)}, but got {instance} instead')

@dataclass
class SpectConfig:
    sample_rate: int = 16000  
    window_size: float = .02  
    window_stride: float = .01 
    window: SpectrogramWindow = SpectrogramWindow.hamming  

    def __post_init__(self):
        validate_type(self.sample_rate, int, "sample_rate", "SpectConfig")
        validate_type(self.window_size, float, "window_size", "SpectConfig")
        validate_type(self.window_stride, float, "window_stride", "SpectConfig")
        validate_enum(self.window, SpectrogramWindow, "window", "SpectConfig")

@dataclass
class AugmentationConfig:
    speed_volume_perturb: bool = False 
    spec_augment: bool = False  
    noise_dir: str =''  
    noise_prob: float = 0.4  
    noise_min: float = 0.0  
    noise_max: float = 0.5  

    def __post_init__(self):
        validate_type(self.speed_volume_perturb, bool, "speed_volume_perturb", "AugmentationConfig")
        validate_type(self.spec_augment, bool, "spec_augment", "AugmentationConfig")
        validate_type(self.noise_dir, str, "noise_dir", "AugmentationConfig")
        validate_type(self.noise_prob, float, "noise_prob", "AugmentationConfig")
        if self.noise_prob < 0 or self.noise_prob > 1:
            raise ValueError(f"Error: AugmentationConfig.noise_prob should be between 0 and 1, but got {self.noise_prob} instead")
        validate_type(self.noise_min, float, "noise_min", "AugmentationConfig")
        validate_type(self.noise_max, float, "noise_max", "AugmentationConfig")
        if self.noise_min < 0 or self.noise_min > 1:
            raise ValueError(f"Error: AugmentationConfig.noise_min should be between 0 and 1, but got {self.noise_min} instead")
        if self.noise_max < 0 or self.noise_max > 1:
            raise ValueError(f"Error: AugmentationConfig.noise_max should be between 0 and 1, but got {self.noise_max} instead")

@dataclass
class DataConfig:
    train_path: str = 'data/train_manifest.csv'
    val_path: str = 'data/val_manifest.csv'
    batch_size: int = 64  
    num_workers: int = 4 
    labels_path: str = 'labels.json' 
    spect: SpectConfig = SpectConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    prepare_data_per_node: bool = True

@dataclass
class BiDirectionalConfig:
    rnn_type: RNNType = RNNType.lstm  
    hidden_size: int = 1024  
    hidden_layers: int = 5 

    def __post_init__(self):
        validate_enum(self.rnn_type, RNNType, "rnn_type", "BiDirectionalConfig")
        validate_type(self.hidden_size, int, "hidden_size", "BiDirectionalConfig")
        validate_type(self.hidden_layers, int, "hidden_layers", "BiDirectionalConfig")

@dataclass
class UniDirectionalConfig(BiDirectionalConfig):
    lookahead_context: int = 20  

    def __post_init__(self):
        validate_type(self.lookahead_context, int, "lookahead_context", "UniDirectionalConfig")

@dataclass
class OptimConfig:
    learning_rate: float = 1.5e-4  
    learning_anneal: float = 0.99 
    weight_decay: float = 1e-5  

    def __post_init__(self):
        validate_type(self.learning_rate, float, "learning_rate", "OptimConfig")
        validate_type(self.learning_anneal, float, "learning_anneal", "OptimConfig")
        validate_type(self.weight_decay, float, "weight_decay", "OptimConfig")

@dataclass
class SGDConfig(OptimConfig):   
    momentum: float = 0.9  

    def __post_init__(self):
        validate_type(self.momentum, float, "momentum", "SGDConfig")

@dataclass
class AdamConfig(OptimConfig):   
    eps: float = 1e-8  
    betas: tuple = (0.9, 0.999)  

    def __post_init__(self):
        validate_type(self.eps, float, "eps", "AdamConfig")
        validate_type(self.betas, tuple, "betas", "AdamConfig")
        if len(self.betas) != 2:
            raise ValueError(f"Error: AdamConfig.betas should be a tuple of length 2, but got a tuple of length {len(self.betas)} instead")
        validate_type(self.betas[0], float, "betas[0]", "AdamConfig")
        validate_type(self.betas[1], float, "betas[1]", "AdamConfig")

@dataclass
class DeepSpeechTrainerConf(TrainerConf):
    callbacks: Any = MISSING

@dataclass
class DeepSpeechConfig:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    optim: Any = MISSING
    model: Any = MISSING
    checkpoint: ModelCheckpointConf = MISSING
    trainer: DeepSpeechTrainerConf = DeepSpeechTrainerConf()
    data: DataConfig = DataConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    seed: int = 123456  
    load_auto_checkpoint: bool = False

    def __post_init__(self):
        validate_type(self.defaults, list, "defaults", "DeepSpeechConfig")
        for default in self.defaults:
            if not isinstance(default, dict):
                raise TypeError(f"Error: DeepSpeechConfig.defaults should be a list of dictionaries, but got a list containing {type(default).__name__} instead")
            elif len(default) != 1:
                raise ValueError(f"Error: Every element in DeepSpeechConfig.defaults should be a dictionary of length 1, but got a dictionary of length {len(default)} instead")
            elif "optim" not in default and "model" not in default and "checkpoint" not in default:
                raise ValueError(f"Error: Every element in DeepSpeechConfig.defaults should either contain the key 'optim', 'model', or 'checkpoint', but got {default.keys()} instead")
            elif "optim" in default:
                if not isinstance(default["optim"], str):
                    raise TypeError(f"Error: The value in DeepSpeechConfig.defaults dictionary with the key 'optim' should be a string, but got a {type(default['optim']).__name__} instead")
            elif "model" in default:
                if not isinstance(default["model"], str):
                    raise TypeError(f"Error: The value in DeepSpeechConfig.defaults dictionary with the key 'model' should be a string, but got a {type(default['model']).__name__} instead")
            elif "checkpoint" in default:
                if not isinstance(default["checkpoint"], str):
                    raise TypeError(f"Error: The value in DeepSpeechConfig.defaults dictionary with the key 'checkpoint' should be a string, but got a {type(default['checkpoint']).__name__} instead")
        validate_type(self.optim, str, "optim", "DeepSpeechConfig")
        validate_type(self.model, str, "model", "DeepSpeechConfig")
        validate_type(self.checkpoint, ModelCheckpointConf, "checkpoint", "DeepSpeechConfig")
        validate_type(self.trainer, DeepSpeechTrainerConf, "trainer", "DeepSpeechConfig")
        validate_type(self.data, DataConfig, "data", "DeepSpeechConfig")
        validate_type(self.augmentation, AugmentationConfig, "augmentation", "DeepSpeechConfig")
        validate_type(self.seed, int, "seed", "DeepSpeechConfig")
        validate_type(self.load_auto_checkpoint, bool, "load_auto_checkpoint", "DeepSpeechConfig")