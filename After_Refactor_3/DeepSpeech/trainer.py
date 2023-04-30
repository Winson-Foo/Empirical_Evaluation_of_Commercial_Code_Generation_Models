from dataclasses import asdict

import pytorch_lightning as pl

from deepspeech_pytorch import LightningDeepSpeech, transcribe
from deepspeech_pytorch.configs.lightning_config import ModelCheckpointConf
from deepspeech_pytorch.training import evaluate


def train_model(cfg, train_path, val_path, checkpoint_dir):
    trainer_conf = {
        'max_epochs': cfg.max_epochs,
        'precision': cfg.precision,
        'gpus': cfg.gpus,
        'limit_train_batches': cfg.limit_train_batches,
        'limit_val_batches': cfg.limit_val_batches,
        'checkpoint_callback': ModelCheckpointConf(
            dirpath=checkpoint_dir,
            save_last=True,
            verbose=True
        ),
    }
    trainer = pl.Trainer(**trainer_conf)

    model_conf = {}
    if cfg.model_config:
        model_conf.update(cfg.model_config)

    deepspeech_cfg = {
        'data': {'train_path': train_path, 'val_path': val_path, 'batch_size': cfg.batch_size},
        'optim': {},
        'model': model_conf,
        'trainer': asdict(cfg)
    }
    model = LightningDeepSpeech(deepspeech_cfg)
    trainer.fit(model)
    return model


def evaluate_model(cfg, test_path, model):
    eval_conf = {
        'model': {
            'model_path': model.checkpoint_callback.best_model_path,
            'cuda': cfg.cuda,
            'precision': cfg.precision
        },
        'lm': {},
        'test_path': test_path
    }
    lm_configs = create_lm_configs()
    for lm_config in lm_configs:
        eval_conf['lm'] = lm_config
        evaluate(EvalConfig(**eval_conf))


def transcribe_audio(cfg, test_path, model):
    file_path = select_test_file(test_path)
    transcribe_cfg = {
        'model': {
            'model_path': model.checkpoint_callback.best_model_path,
            'cuda': cfg.cuda,
            'precision': cfg.precision
        },
        'lm': {},
        'audio_path': file_path
    }
    lm_configs = create_lm_configs()
    for lm_config in lm_configs:
        transcribe_cfg['lm'] = lm_config
        transcribe(TranscribeConfig(**transcribe_cfg))