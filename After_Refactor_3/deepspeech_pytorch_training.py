import json

import hydra
from hydra.utils import to_absolute_path
from pytorch_lightning import seed_everything

from deepspeech_pytorch.checkpoint import FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech


def load_labels(labels_path):
    with open(to_absolute_path(labels_path)) as label_file:
        return json.load(label_file)


def configure_checkpoint(cfg):
    if cfg.trainer.enable_checkpointing:
        checkpoint_callback = FileCheckpointHandler(
            cfg=cfg.checkpoint
        )
        if cfg.load_auto_checkpoint:
            resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
            if resume_from_checkpoint:
                cfg.trainer.resume_from_checkpoint = resume_from_checkpoint
        return checkpoint_callback
    else:
        return None


def train_model(cfg):
    seed_everything(cfg.seed)

    labels = load_labels(cfg.data.labels_path)
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

    checkpoint_callback = configure_checkpoint(cfg)

    trainer = hydra.utils.instantiate(
        config=cfg.trainer,
        replace_sampler_ddp=False,
        callbacks=[checkpoint_callback] if checkpoint_callback else None,
    )
    trainer.fit(model, data_loader)


def train(cfg: DeepSpeechConfig):
    train_model(cfg)