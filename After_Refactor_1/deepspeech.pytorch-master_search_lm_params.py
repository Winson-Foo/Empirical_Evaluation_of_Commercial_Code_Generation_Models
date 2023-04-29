# main.py

import hydra
from hydra.core.config_store import ConfigStore

from optimizer import Optimizer
from configs import OptimizerConfig


cs = ConfigStore.instance()
cs.store(name="config", node=OptimizerConfig)


@hydra.main(config_name="config")
def main(cfg: OptimizerConfig) -> None:
    optimizer = Optimizer(
        model_path=cfg.model_path,
        test_path=cfg.test_path,
        lm_path=cfg.lm_path,
        is_character_based=cfg.is_character_based,
        beam_width=cfg.beam_width,
        alpha_from=cfg.alpha_from,
        alpha_to=cfg.alpha_to,
        beta_from=cfg.beta_from,
        beta_to=cfg.beta_to,
        n_trials=cfg.n_trials,
        n_jobs=cfg.n_jobs,
        precision=cfg.precision,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        spect_cfg=cfg.spect_cfg,
    )
    optimizer.run()


if __name__ == "__main__":
    main()
