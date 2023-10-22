import json
from typing import Dict

import hydra
from hydra.utils import to_absolute_path
from pytorch_lightning import seed_everything

from deepspeech_pytorch.checkpoint import FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech

LABELS_PATH = "data.labels_path"


def load_labels(cfg: DeepSpeechConfig) -> Dict:
    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)
    return labels


def configure_checkpoint(cfg: DeepSpeechConfig) -> FileCheckpointHandler:
    if not cfg.trainer.enable_checkpointing:
        return None

    checkpoint_callback = FileCheckpointHandler(
        cfg=cfg.checkpoint
    )

    if cfg.load_auto_checkpoint:
        resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
        if resume_from_checkpoint:
            cfg.trainer.resume_from_checkpoint = resume_from_checkpoint

    return checkpoint_callback


def train(cfg: DeepSpeechConfig) -> None:
    seed_everything(cfg.seed)

    labels = load_labels(cfg)
    checkpoint_callback = configure_checkpoint(cfg)

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

    callbacks = [checkpoint_callback] if checkpoint_callback else None
    trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        replace_sampler_ddp=False,
        callbacks=callbacks,
    )
    trainer.fit(model, data_loader)