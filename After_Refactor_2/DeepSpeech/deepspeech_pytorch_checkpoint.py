import os
from typing import Optional, Union
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load

from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf


class CheckpointHandler(ModelCheckpoint):
    def __init__(self, cfg: ModelCheckpointConf):
        super().__init__(**cfg.to_dict())

    def find_latest_checkpoint(self) -> Optional[Union[str, Path]]:
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        If there are no checkpoints, returns None.
        :return: The latest checkpoint path, or None if no checkpoints are found.
        """
        paths = list(Path(self.dirpath).rglob("*"))
        if paths:
            paths.sort(key=os.path.getctime)
            latest_checkpoint_path = paths[-1]
            return latest_checkpoint_path
        else:
            return None

    def on_load_checkpoint(self, checkpoint_path: str) -> dict:
        try:
            return pl_load(checkpoint_path)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return {}


class FileCheckpointHandler(CheckpointHandler):
    def find_latest_checkpoint(self) -> Optional[Union[str, Path]]:
        """
        Finds the latest checkpoint in a folder based on the timestamp of the file.
        If there are no checkpoints, returns None.
        :return: The latest checkpoint path, or None if no checkpoints are found.
        """
        latest_checkpoint_path = super().find_latest_checkpoint()
        if latest_checkpoint_path is not None:
            return str(latest_checkpoint_path)
        else:
            return None