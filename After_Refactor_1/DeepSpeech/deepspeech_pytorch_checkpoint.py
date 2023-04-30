import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf


class CheckpointHandler(pl.callbacks.ModelCheckpoint):
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
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        If there are no checkpoints, returns None.
        """
        paths = list(Path(self.dirpath).rglob('*'))
        if not paths:
            return None
        latest_checkpoint_path = max(paths, key=os.path.getctime)
        return latest_checkpoint_path


class FileCheckpointHandler(CheckpointHandler):
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        If there are no checkpoints, returns None.
        """
        latest_checkpoint_path = super().find_latest_checkpoint()
        return latest_checkpoint_path