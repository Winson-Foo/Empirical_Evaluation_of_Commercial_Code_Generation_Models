from dataclasses import dataclass, field
from typing import Any, Optional, Union

@dataclass
class ModelCheckpointConf:
    _target_: str = "pytorch_lightning.callbacks.ModelCheckpoint"
    filepath: Optional[str] = None
    monitor: Optional[str] = None
    verbose: bool = False
    save_last: Optional[bool] = None
    save_top_k: Optional[int] = 1
    save_weights_only: bool = False
    mode: str = "min"
    dirpath: Optional[Union[str, Path]] = None  
    filename: Optional[str] = None
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[str] = None
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None


@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.trainer.Trainer"
    logger: Union[bool, LightningLoggerBase, Iterable[Union[bool, LightningLoggerBase]]] = True
    enable_checkpointing: bool = True
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    callbacks: Optional[Any] = None
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Optional[Union[int, str, List[int]]] = None 
    auto_select_gpus: bool = False
    tpu_cores: Optional[Union[int, str, List[int]]] = None 
    overfit_batches: float = 0.0 
    track_grad_norm: Union[float, int, str] = -1 
    check_val_every_n_epoch: int = 1
    fast_dev_run: Union[bool, int] = False
    accumulate_grad_batches: Any = 1  
    max_epochs: int = 1000
    min_epochs: int = 1
    limit_train_batches: Union[float, int] = 1.0  
    limit_val_batches: Union[float, int] = 1.0  
    limit_test_batches: Union[float, int] = 1.0  
    val_check_interval: Union[float, int] = 1  
    log_every_n_steps: int = 50
    accelerator: Optional[Union[str, Accelerator]] = None 
    sync_batchnorm: bool = False
    precision: int = 32
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    resume_from_checkpoint: Optional[Union[str, Path]] = None 
    profiler: Optional[Union[BaseProfiler, bool, str]] = None 
    benchmark: bool = False
    deterministic: bool = False
    auto_lr_find: Union[bool, str] = False  
    replace_sampler_ddp: bool = True
    detect_anomaly: bool = False
    auto_scale_batch_size: Union[bool, str] = False  
    plugins: Optional[Union[str, List[str]]] = None  
    amp_backend: str = "native"
    amp_level: Optional[str] = None
    move_metrics_to_cpu: bool = False
    gradient_clip_algorithm: Optional[str] = None
    devices: Optional[Any] = None 
    ipus: Optional[int] = None  
    enable_progress_bar: bool = True
    max_time: Optional[str] = None
    limit_predict_batches: float = 1.0 
    strategy: Optional[str] = None 
    enable_model_summary: bool = True
    reload_dataloaders_every_n_epochs: int = 0
    multiple_trainloader_mode: str = "max_size_cycle"