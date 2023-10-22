from typing import List
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore

import optuna
import torch

from deepspeech_pytorch.configs.train_config import SpectConfig
from deepspeech_pytorch.decoder import BeamCTCDecoder, GreedyDecoder
from deepspeech_pytorch.loader.data_loader import AudioDataLoader, SpectrogramDataset
from deepspeech_pytorch.utils import load_model
from deepspeech_pytorch.validation import run_evaluation


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


def load_data(cfg: OptimizerConfig) -> Tuple[AudioDataLoader, BeamCTCDecoder, GreedyDecoder, List[str]]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device, hydra.utils.to_absolute_path(cfg.model_path))
    ckpt = torch.load(hydra.utils.to_absolute_path(cfg.model_path), map_location=device)
    labels = ckpt['hyper_parameters']['labels']

    decoder = BeamCTCDecoder(
        labels=labels,
        lm_path=hydra.utils.to_absolute_path(cfg.lm_path),
        beam_width=cfg.beam_width,
        num_processes=cfg.num_workers,
        blank_index=labels.index('_')
    )
    target_decoder = GreedyDecoder(labels=labels, blank_index=labels.index('_'))

    test_dataset = SpectrogramDataset(
        audio_conf=cfg.spect_cfg,
        input_path=hydra.utils.to_absolute_path(cfg.test_path),
        labels=labels,
        normalize=True
    )
    test_loader = AudioDataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    
    return test_loader, decoder, target_decoder, labels


def evaluate(trial: optuna.Trial, cfg: OptimizerConfig, data: Tuple[AudioDataLoader, BeamCTCDecoder, GreedyDecoder, List[str]]) -> float:
    alpha = trial.suggest_uniform('alpha', cfg.alpha_from, cfg.alpha_to)
    beta = trial.suggest_uniform('beta', cfg.beta_from, cfg.beta_to)
    data[1]._decoder.reset_params(alpha, beta)

    wer, cer = run_evaluation(
        test_loader=data[0],
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        model=data[1],
        decoder=data[2],
        target_decoder=data[2],
        precision=cfg.precision
    )
    return cer if cfg.is_character_based else wer


@hydra.main(config_name="config")
def main(cfg: OptimizerConfig) -> None:
    test_loader, decoder, target_decoder, labels = load_data(cfg)

    study = optuna.create_study()
    study.optimize(lambda trial: evaluate(trial, cfg, (test_loader, decoder, target_decoder, labels)), n_trials=cfg.n_trials, n_jobs=cfg.n_jobs, show_progress_bar=True)

    print(f"Best Params\n"
          f"alpha: {study.best_params['alpha']}\n"
          f"beta: {study.best_params['beta']}\n"
          f"{'cer' if cfg.is_character_based else 'wer'}: {study.best_value}")


if __name__ == "__main__":
    main()