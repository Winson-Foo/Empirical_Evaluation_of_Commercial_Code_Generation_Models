from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Tuple, Iterable

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.profiler import BaseProfiler
from pytorch_lightning.utilities.accelerators import Accelerator


@dataclass
class ModelCheckpointConf:
    target: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    filepath: Union[str, Path, None] = None
    monitor: Union[str, None] = None
    verbose: bool = False
    save_last: Union[bool, None] = None
    save_top_k: int = 1
    save_weights_only: bool = False
    mode: str = "min"
    dirpath: Union[str, Path, None] = None
    filename: Union[str, None] = None
    auto_insert_metric_name: bool = True
    every_n_train_steps: Union[int, None] = None
    train_time_interval: Union[str, None] = None
    every_n_epochs: Union[int, None] = None
    save_on_train_epoch_end: Union[bool, None] = None


@dataclass
class TrainerConf:
    target: str = "pytorch_lightning.trainer.Trainer"
    logger: Union[bool, LightningLoggerBase, Iterable[LightningLoggerBase]] = True
    enable_checkpointing: bool = True
    default_root_dir: Union[str, None] = None
    gradient_clip_val: float = 0
    callbacks: Union[None, List[Callback]] = None
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Union[int, str, List[int], None] = None
    auto_select_gpus: bool = False
    tpu_cores: Union[int, str, List[int], None] = None
    overfit_batches: Union[int, float] = 0.0
    track_grad_norm: Union[int, float, str] = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: Union[int, bool] = False
    accumulate_grad_batches: Union[int, Dict[int, int], List[List[int]]] = 1
    max_epochs: int = 1000
    min_epochs: int = 1
    limit_train_batches: Union[int, float] = 1.0
    limit_val_batches: Union[int, float] = 1.0
    limit_test_batches: Union[int, float] = 1.0
    val_check_interval: Union[int, float] = 1.0
    log_every_n_steps: int = 50
    accelerator: Union[None, str, Accelerator] = None
    sync_batchnorm: bool = False
    precision: int = 32
    weights_save_path: Union[str, None] = None
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Union[str, Path, None] = None
    profiler: Union[bool, str, BaseProfiler, None] = None
    benchmark: bool = False
    deterministic: bool = False
    auto_lr_find: Union[bool, str] = False
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: Union[bool, str] = False
    plugins: Union[List[str], str, None] = None
    amp_backend: str = "native"
    amp_level: Union[str, int, None] = None
    move_metrics_to_cpu: bool = False
    gradient_clip_algorithm: Union[str, None] = None
    devices: Union[None, Tuple[str], List[Tuple[str]]] = None
    ipus: Union[int, None] = None
    enable_progress_bar: bool = True
    max_time: Union[str, None] = None
    limit_predict_batches: float = 1.0
    strategy: Union[str, None] = None
    enable_model_summary: bool = True
    reload_dataloaders_every_n_epochs: int = 0
    multiple_trainloader_mode: str = "max_size_cycle"