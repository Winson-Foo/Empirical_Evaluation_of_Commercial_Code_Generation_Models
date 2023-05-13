import os
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint

from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf


class CheckpointHandler(ModelCheckpoint):

    def __init__(self, cfg: ModelCheckpointConf):
        try:
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
        except Exception as e:
            print(f"Error while initializing CheckpointHandler: {e}")

    def find_latest_checkpoint(self):
        try:
            raise NotImplementedError
        except Exception as e:
            print(f"Error while finding latest checkpoint: {e}")
            return None


class FileCheckpointHandler(CheckpointHandler):

    def find_latest_checkpoint(self):
        try:
            paths = list(Path(self.dirpath).rglob('*'))
            if paths:
                paths.sort(key=os.path.getctime)
                latest_checkpoint_path = paths[-1]
                return latest_checkpoint_path
            else:
                return None
        except Exception as e:
            print(f"Error while finding latest checkpoint: {e}")
            return None