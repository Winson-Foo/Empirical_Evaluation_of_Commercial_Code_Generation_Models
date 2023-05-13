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
        # Seed for reproducibility
        seed_everything(cfg.seed)
        
        # Load labels from file
        with open(to_absolute_path(cfg.data.labels_path)) as label_file:
            labels = json.load(label_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    # Check if checkpointing is enabled
    checkpoint_callback = None
    if cfg.trainer.enable_checkpointing:
        try:
            # Initialize checkpoint handler
            checkpoint_callback = FileCheckpointHandler(
                cfg=cfg.checkpoint
            )
            
            # Try to load latest checkpoint
            if cfg.load_auto_checkpoint:
                resume_from_checkpoint = checkpoint_callback.find_latest_checkpoint()
                if resume_from_checkpoint:
                    cfg.trainer.resume_from_checkpoint = resume_from_checkpoint
        except Exception as e:
            print(f"Error setting up checkpointing: {e}")
            return
    
    try:
        # Initialize data loader
        data_loader = DeepSpeechDataModule(
            labels=labels,
            data_cfg=cfg.data,
            normalize=True,
        )
        
        # Initialize model
        model = DeepSpeech(
            labels=labels,
            model_cfg=cfg.model,
            optim_cfg=cfg.optim,
            precision=cfg.trainer.precision,
            spect_cfg=cfg.data.spect
        )
        
        # Initialize trainer
        callbacks = [checkpoint_callback] if checkpoint_callback else None
        trainer = hydra.utils.instantiate(
            config=cfg.trainer,
            replace_sampler_ddp=False,
            callbacks=callbacks
        )
        
        # Train model
        trainer.fit(model, data_loader)
    except Exception as e:
        print(f"Error training model: {e}")
        return