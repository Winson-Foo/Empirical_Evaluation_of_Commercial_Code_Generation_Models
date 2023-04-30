# configs.py

from dataclasses import dataclass

from deepspeech_pytorch.configs.train_config import SpectConfig


@dataclass
class OptimizerConfig:
    model_path: str = ''
    test_path: str = ''  # Path to test manifest or csv
    is_character_based: bool = True  # Use CER or WER for finding optimal parameters
    lm_path: str = ''
    beam_width: int = 10
    alpha_from: float = 0.0
    alpha_to: float = 3.0
    beta_from: float = 0.0
    beta_to: float = 1.0
    n_trials: int = 500  # Number of trials for optuna
    n_jobs: int = 2      # Number of parallel jobs for optuna
    precision: int = 16
    batch_size: int = 1   # For dataloader
    num_workers: int = 1  # For dataloader
    spect_cfg: SpectConfig = SpectConfig()