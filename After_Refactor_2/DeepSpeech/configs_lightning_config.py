from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelCheckpointConf:
    target: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    filepath: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: Optional[int] = 1
    save_weights_only: bool = False
    mode: str = "min"
    dirpath: Optional[str] = None
    filename: Optional[str] = None
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[str] = None
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None
