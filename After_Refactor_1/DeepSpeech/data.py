import os
import json
from dataclasses import dataclass
from deepspeech_pytorch.enums import DecoderType
from pathlib import Path
from data.an4 import download_an4

@dataclass
class DatasetConfig:
    target_dir: str = ''
    manifest_dir: str = ''
    min_duration: float = 0
    max_duration: float = 15
    val_fraction: float = 0.1
    sample_rate: int = 16000
    num_workers: int = 4
    
def download_an4_data(cfg: DatasetConfig, folders: bool = False):
    download_an4(
        target_dir=cfg.target_dir,
        manifest_dir=cfg.manifest_dir,
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
        num_workers=cfg.num_workers
    )

    if folders:
        train_path = os.path.join(cfg.target_dir, 'train/')
        val_path = os.path.join(cfg.target_dir, 'val/')
        test_path = os.path.join(cfg.target_dir, 'test/')
    else:
        train_path = os.path.join(cfg.manifest_dir, 'an4_train_manifest.json')
        val_path = os.path.join(cfg.manifest_dir, 'an4_val_manifest.json')
        test_path = os.path.join(cfg.manifest_dir, 'an4_test_manifest.json')

    assert os.path.exists(train_path)
    assert os.path.exists(val_path)
    assert os.path.exists(test_path)

    return train_path, val_path, test_path