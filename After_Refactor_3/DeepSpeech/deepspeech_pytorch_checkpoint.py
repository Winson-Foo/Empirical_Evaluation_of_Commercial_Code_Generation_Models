import os
from pathlib import Path
from typing import Optional

from pytorch_lightning.callbacks import ModelCheckpoint
from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf


class CustomModelCheckpoint(ModelCheckpoint):
    """
    Custom wrapper for ModelCheckpoint to provide additional functionality.
    """

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

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Returns the file path of the latest checkpoint in the checkpoint directory.
        Returns None if no checkpoints are found in the directory.
        """
        paths = list(Path(self.dirpath).rglob("*"))
        if paths:
            try:
                paths.sort(key=os.path.getctime)
                latest_checkpoint_path = str(paths[-1])
                return latest_checkpoint_path
            except Exception as e:
                print(f"Error in getting latest checkpoint path: {e}")
                return None
        else:
            return None