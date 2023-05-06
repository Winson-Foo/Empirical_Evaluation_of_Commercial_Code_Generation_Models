import json

import hydra
from hydra.utils import to_absolute_path
from pytorch_lightning import seed_everything

from deepspeech_pytorch.checkpoint import FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.model import DeepSpeech

def train(cfg: DeepSpeechConfig):
    try:
        seed_everything(cfg.seed)
        label_file = open(to_absolute_path(cfg.data.labels_path))
        labels = json.load(label_file)
        label_file.close()

        if cfg.trainer.enable_checkpointing:
            checkpoint_callback = FileCheckpointHandler(cfg=cfg.checkpoint)
            
            if cfg.load_auto_checkpoint:
                resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
                if resume_from_checkpoint:
                    cfg.trainer.resume_from_checkpoint = resume_from_checkpoint

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

        callbacks = [checkpoint_callback] if cfg.trainer.enable_checkpointing else None
        
        trainer = hydra.utils.instantiate(config=cfg.trainer, replace_sampler_ddp=False, callbacks=callbacks)
        trainer.fit(model, data_loader)
    except Exception as e:
        print("Error during training: ", e)
        raise e
    finally:
        if 'label_file' in locals():
            label_file.close()