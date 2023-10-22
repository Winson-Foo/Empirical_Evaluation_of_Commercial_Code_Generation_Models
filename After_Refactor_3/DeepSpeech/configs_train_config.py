# config.py
from dataclasses import dataclass, field
from typing import Any, List
from .data import DataConfig, AugmentationConfig, SpectConfig
from .model import BiDirectionalConfig, UniDirectionalConfig, OptimConfig, AdamConfig, DeepSpeechTrainerConf

@dataclass
class SpeechRecognitionConfig:
    seed: int = 123456
    load_auto_checkpoint: bool = False
    
    data: DataConfig = DataConfig()
    data.train_path: str = 'data/train_manifest.csv'
    data.val_path: str = 'data/val_manifest.csv'
    data.batch_size: int = 64
    data.num_workers: int = 4
    data.labels_path: str = 'labels.json'
    data.spect: SpectConfig = SpectConfig()
    data.augmentation: AugmentationConfig = AugmentationConfig()
    
    model: Any = field(default_factory=lambda: ModelConfig(
        optim=AdamConfig(),
        model_type='bidirectional',
        bidirectional_config=BiDirectionalConfig(),
        unidirectional_config=UniDirectionalConfig()
    ))
    
    trainer: DeepSpeechTrainerConf = DeepSpeechTrainerConf(
        max_epochs=50,
        gpus=1,
        callbacks=[]
    )