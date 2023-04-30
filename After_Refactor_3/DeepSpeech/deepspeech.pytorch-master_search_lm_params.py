from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
import optuna
import torch

from deepspeech_pytorch.configs.train_config import SpectConfig
from deepspeech_pytorch.decoder import BeamCTCDecoder, GreedyDecoder
from deepspeech_pytorch.loader.data_loader import (
    AudioDataLoader,
    SpectrogramDataset,
)
from deepspeech_pytorch.utils import load_model
from deepspeech_pytorch.validation import run_evaluation

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BLANK_INDEX = '_'

# Enumerations
class ScoreType:
    CHARACTER = 'cer'
    WORD = 'wer'

# Data Classes
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


# Config Store
cs = ConfigStore.instance()
cs.store(name="config", node=OptimizerConfig)


# Objective Function
class Objective:
    def __init__(self, config):
        self.config = config

        self.model = load_model(DEVICE, hydra.utils.to_absolute_path(self.config.model_path))
        self.ckpt = torch.load(
            hydra.utils.to_absolute_path(self.config.model_path), map_location=DEVICE
        )
        self.labels = self.ckpt['hyper_parameters']['labels']
        self.decoder = BeamCTCDecoder(
            labels=self.labels,
            lm_path=hydra.utils.to_absolute_path(self.config.lm_path),
            beam_width=self.config.beam_width,
            num_processes=self.config.num_workers,
            blank_index=BLANK_INDEX,
        )
        self.target_decoder = GreedyDecoder(labels=self.labels, blank_index=BLANK_INDEX)

        test_dataset = SpectrogramDataset(
            audio_conf=self.config.spect_cfg,
            input_path=hydra.utils.to_absolute_path(self.config.test_path),
            labels=self.labels,
            normalize=True,
        )
        self.test_loader = AudioDataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )

    def __call__(self, trial):
        alpha = trial.suggest_uniform('alpha', self.config.alpha_from, self.config.alpha_to)
        beta = trial.suggest_uniform('beta', self.config.beta_from, self.config.beta_to)
        self.decoder._decoder.reset_params(alpha, beta)

        wer, cer = self._evaluate_model()

        return cer if self.config.is_character_based else wer

    def _evaluate_model(self):
        return run_evaluation(
            test_loader=self.test_loader,
            device=DEVICE,
            model=self.model,
            decoder=self.decoder,
            target_decoder=self.target_decoder,
            precision=self.config.precision,
        )


# Main Function
@hydra.main(config_name="config")
def main(config: OptimizerConfig) -> None:
    study = optuna.create_study()
    study.optimize(
        Objective(config),
        n_trials=config.n_trials,
        n_jobs=config.n_jobs,
        show_progress_bar=True,
    )
    self._print_results(study.best_params, study.best_value, config.is_character_based)

    return None

def _print_results(best_params, best_score, is_character_based):
    score_type = ScoreType.CHARACTER if is_character_based else ScoreType.WORD

    print(f"Best Params\n"
          f"alpha: {best_params['alpha']}\n"
          f"beta: {best_params['beta']}\n"
          f"{score_type}: {best_score}")

if __name__ == "__main__":
    main()