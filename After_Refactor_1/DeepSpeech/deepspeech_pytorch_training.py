import json
from typing import List, Dict, Optional

import hydra
from hydra.utils import to_absolute_path
from pytorch_lightning import seed_everything, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from deepspeech_pytorch.checkpoint import FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech


def load_labels(labels_path: str) -> Dict[str, str]:
    with open(to_absolute_path(labels_path), "r") as label_file:
        labels = json.load(label_file)
    return labels


def find_checkpoint(cfg: DeepSpeechConfig) -> Optional[str]:
    if not cfg.trainer.enable_checkpointing:
        return None
    checkpoint_handler = FileCheckpointHandler(cfg.checkpoint)
    if cfg.load_auto_checkpoint:
        return checkpoint_handler.find_latest_checkpoint()
    return None


def train_model(model: LightningModule, data_loader: DeepSpeechDataModule, trainer: Trainer,
                checkpoint_callback: Optional[Callback] = None):
    trainer.fit(model, data_loader, callbacks=[checkpoint_callback] if checkpoint_callback else None)


def train(cfg: DeepSpeechConfig):
    seed_everything(cfg.seed)

    labels = load_labels(cfg.data.labels_path)

    checkpoint = find_checkpoint(cfg)
    if checkpoint:
        cfg.trainer.resume_from_checkpoint = checkpoint
        checkpoint_callback = FileCheckpointHandler(cfg=cfg.checkpoint)
    else:
        checkpoint_callback = None

    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
    )

    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.trainer.precision,
        spect_cfg=cfg.data.spect
    )

    trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        replace_sampler_ddp=False,
    )

    train_model(model, data_loader, trainer, checkpoint_callback)
