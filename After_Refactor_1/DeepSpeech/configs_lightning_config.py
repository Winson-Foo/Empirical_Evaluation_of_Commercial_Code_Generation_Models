from typing import Any, Dict, List, Union
from dataclasses import dataclass

@dataclass
class ModelCheckpointConfig:
    target_cls: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    filepath: str = ""
    monitor_metric: str = ""
    verbose: bool = False
    save_last: bool = False
    save_top_k: int = 1
    save_weights_only: bool = False
    mode: str = "min"
    dirpath: Union[str, None] = None
    filename: str = ""
    auto_insert_metric_name: bool = True
    every_n_train_steps: int = -1
    train_time_interval: Union[str, None] = None
    every_n_epochs: int = -1
    save_on_train_epoch_end: bool = False


@dataclass
class LoggerConfig:
    target_cls: str
    parameters: Dict[str, Any]


@dataclass
class TrainerConfig:
    target_cls: str = "pytorch_lightning.trainer.Trainer"
    logger: Union[LoggerConfig, List[LoggerConfig], bool] = True
    checkpoint_config: Union[ModelCheckpointConfig, Dict[str, Any]] = None
    default_root_dir: Union[str, None] = None
    gradient_clip_val: float = 0.0
    callbacks: List[Union[str, Dict[str, Any]]] = None
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Union[List[int], str] = []
    auto_select_gpus: bool = False
    tpu_cores: Union[List[int], str] = []
    overfit_batches: Union[float, int] = 0.0
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
    accelerator: Union[str, Any] = None
    sync_batchnorm: bool = False
    precision: int = 32
    weights_save_path: Union[str, None] = None
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Union[str, None] = None
    profiler: Union[bool, str, Any] = None
    benchmark: bool = False
    deterministic: bool = False
    auto_lr_find: Union[str, bool] = False
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: Union[str, bool] = False
    plugins: List[str] = None
    amp_backend: str = "native"
    amp_level: Union[str, int, None] = None
    move_metrics_to_cpu: bool = False
    gradient_clip_algorithm: Union[str, None] = None
    devices: Union[List[int], str] = []
    ipus: Union[int, None] = None
    enable_progress_bar: bool = True
    max_time: Union[str, None] = None
    limit_predict_batches: float = 1.0
    strategy: Union[str, None] = None
    enable_model_summary: bool = True
    reload_dataloaders_every_n_epochs: int = 0
    multiple_trainloader_mode: str = "max_size_cycle"
