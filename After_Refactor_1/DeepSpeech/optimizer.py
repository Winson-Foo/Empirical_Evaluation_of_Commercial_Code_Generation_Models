# optimizer.py

from typing import List

import optuna
import torch

from deepspeech_pytorch.decoder import BeamCTCDecoder, GreedyDecoder
from deepspeech_pytorch.loader.data_loader import AudioDataLoader, SpectrogramDataset
from deepspeech_pytorch.utils import load_model
from deepspeech_pytorch.validation import run_evaluation


class Optimizer:
    def __init__(self, model_path: str, test_path: str, lm_path: str, is_character_based: bool = True, beam_width: int = 10,
                 alpha_from: float = 0.0, alpha_to: float = 3.0, beta_from: float = 0.0, beta_to: float = 1.0,
                 n_trials: int = 500, n_jobs: int = 2, precision: int = 16, batch_size: int = 1, num_workers: int = 1,
                 labels: List[str] = None, device: str = 'cuda'):

        self.model_path = model_path
        self.test_path = test_path
        self.lm_path = lm_path
        self.is_character_based = is_character_based
        self.beam_width = beam_width
        self.alpha_from = alpha_from
        self.alpha_to = alpha_to
        self.beta_from = beta_from
        self.beta_to = beta_to
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.precision = precision
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.device = torch.device(device)
        self.model = load_model(self.device, model_path)
        self.ckpt = torch.load(model_path, map_location=self.device)
        if labels is not None:
            self.labels = labels
        else:
            self.labels = self.ckpt['hyper_parameters']['labels']

        self.decoder = BeamCTCDecoder(
            labels=self.labels,
            lm_path=lm_path,
            beam_width=self.beam_width,
            num_processes=self.num_workers,
            blank_index=self.labels.index('_')
        )
        self.target_decoder = GreedyDecoder(
            labels=self.labels,
            blank_index=self.labels.index('_')
        )

        test_dataset = SpectrogramDataset(
            audio_conf=SpectConfig(),
            input_path=self.test_path,
            labels=self.labels,
            normalize=True
        )
        self.test_loader = AudioDataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def run(self):
        study = optuna.create_study()
        study.optimize(self._objective,
                       n_trials=self.n_trials,
                       n_jobs=self.n_jobs,
                       show_progress_bar=True)
        print(f"Best Params\n"
              f"alpha: {study.best_params['alpha']}\n"
              f"beta: {study.best_params['beta']}\n"
              f"{'cer' if self.is_character_based else 'wer'}: {study.best_value}")

    def _objective(self, trial):
        alpha = trial.suggest_uniform('alpha', self.alpha_from, self.alpha_to)
        beta = trial.suggest_uniform('beta', self.beta_from, self.beta_to)
        self.decoder._decoder.reset_params(alpha, beta)

        wer, cer = run_evaluation(
            test_loader=self.test_loader,
            device=self.device,
            model=self.model,
            decoder=self.decoder,
            target_decoder=self.target_decoder,
            precision=self.precision
        )
        return cer if self.is_character_based else wer