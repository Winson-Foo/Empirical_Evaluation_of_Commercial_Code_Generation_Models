import json
import os

import hydra
from hydra.utils import to_absolute_path
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from deepspeech_pytorch.checkpoint import FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech


def train(cfg: DeepSpeechConfig):
    seed_everything(cfg.seed)

    if 'labels_path' not in cfg['data'] or not os.path.isfile(to_absolute_path(cfg.data.labels_path)):
        raise ValueError("Missing or invalid labels file path")

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    checkpoint_callback = None
    if cfg.trainer.enable_checkpointing:
        if 'checkpoint' not in cfg:
            raise ValueError("Missing checkpoint configuration")
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min'
        )
        if cfg.load_auto_checkpoint:
            resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
            if resume_from_checkpoint:
                cfg.trainer.resume_from_checkpoint = resume_from_checkpoint
                
    if 'data' not in cfg or 'spect' not in cfg.data:
        raise ValueError("Missing spectrogram configuration")
        
    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
    )

    if 'model' not in cfg or 'optim' not in cfg:
        raise ValueError("Missing model or optimizer configuration")

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
        callbacks=[checkpoint_callback] if cfg.trainer.enable_checkpointing else None,
    )
    trainer.fit(model, data_loader)