import os
from pathlib import Path
from typing import Optional

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_warn

from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf


class CheckpointHandler(ModelCheckpoint):

    def __init__(self, cfg: ModelCheckpointConf):
        super().__init__(
            dirpath=cfg.dirpath,
            filename=cfg.filename,
            monitor=cfg.monitor,
            verbose=cfg.verbose,
            save_last=cfg.save_last,
            save_top_k=cfg.save_top_k,
            save_weights_only=cfg.save_weights_only,
            mode=cfg.mode,
            auto_insert_metric_name=cfg.auto_insert_metric_name,
            every_n_train_steps=cfg.every_n_train_steps,
            train_time_interval=cfg.train_time_interval,
            every_n_epochs=cfg.every_n_epochs,
            save_on_train_epoch_end=cfg.save_on_train_epoch_end,
        )

    def find_latest_checkpoint(self) -> Optional[Path]:
        paths = list(Path(self.dirpath).rglob('*'))
        if paths:
            try:
                # sort paths by file modification time, latest first
                paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                latest_checkpoint_path = paths[0]
                return latest_checkpoint_path
            except Exception as e:
                rank_zero_warn(f"Unable to sort checkpoint paths: {e}")
                return None
        else:
            return None