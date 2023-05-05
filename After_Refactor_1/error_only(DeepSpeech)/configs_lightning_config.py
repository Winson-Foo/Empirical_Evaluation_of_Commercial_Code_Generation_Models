from dataclasses import dataclass, field
from typing import Any, List, Union, Optional

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
    dirpath: Optional[str] = None 
    filename: Optional[str] = None
    auto_insert_metric_name: bool = True
    every_n_train_steps: Optional[int] = None
    train_time_interval: Optional[str] = None
    every_n_epochs: Optional[int] = None
    save_on_train_epoch_end: Optional[bool] = None
    
@dataclass
class TrainerConf:
    _target_: str = "pytorch_lightning.trainer.Trainer"
    logger: Union[bool, 'LightningLoggerBase', List['LightningLoggerBase']] = True
    enable_checkpointing: bool = True
    default_root_dir: Optional[str] = None 
    gradient_clip_val: Optional[float] = None 
    callbacks: Optional[List[Any]] = None 
    num_nodes: int = 1 
    num_processes: Optional[int] = None 
    gpus: Optional[Union[int, str, List[int]]] = None 
    auto_select_gpus: bool = False 
    tpu_cores: Optional[Union[int, str, List[int]]] = None 
    overfit_batches: Optional[float] = None 
    track_grad_norm: Optional[Union[int, float, str]] = None 
    check_val_every_n_epoch: Optional[int] = None 
    fast_dev_run: Optional[Union[int, bool]] = None 
    accumulate_grad_batches: Optional[Union[int, Dict[int,int], List[list]]] = None 
    max_epochs: Optional[int] = None 
    min_epochs: Optional[int] = None 
    limit_train_batches: Optional[Union[int, float]] = None 
    limit_val_batches: Optional[Union[int, float]] = None 
    limit_test_batches: Optional[Union[int, float]] = None 
    val_check_interval: Optional[Union[float, int]] = None 
    log_every_n_steps: Optional[int] = None
    accelerator: Optional[Union[str, 'Accelerator']] = None
    sync_batchnorm: Optional[bool] = None
    precision: Optional[int] = None 
    weights_save_path: Optional[str] = None 
    num_sanity_val_steps: Optional[int] = None 
    resume_from_checkpoint: Optional[Union[str, Path]] = None 
    profiler: Optional[Union[BaseProfiler, str, bool]] = None 
    benchmark: Optional[bool] = None 
    deterministic: Optional[bool] = None 
    auto_lr_find: Optional[Union[bool, str]] = None 
    replace_sampler_ddp: Optional[bool] = None 
    detect_anomaly: Optional[bool] = None 
    auto_scale_batch_size: Optional[Union[str, bool]] = None 
    plugins: Optional[List[str]] = None 
    amp_backend: Optional[str] = None 
    amp_level: Optional[str] = None 
    move_metrics_to_cpu: Optional[bool] = None 
    gradient_clip_algorithm: Optional[str] = None 
    devices: Optional[List[Union[int, str]]] = None
    ipus: Optional[int] = None 
    enable_progress_bar: Optional[bool] = None 
    max_time: Optional[str] = None 
    limit_predict_batches: Optional[float] = None 
    strategy: Optional[str] = None 
    enable_model_summary: Optional[bool] = None 
    reload_dataloaders_every_n_epochs: Optional[int] = None 
    multiple_trainloader_mode: Optional[str] = None