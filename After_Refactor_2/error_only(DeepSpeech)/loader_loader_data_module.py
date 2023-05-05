import pytorch_lightning as pl
from hydra.utils import to_absolute_path
import torch
from deepspeech_pytorch.configs.train_config import DataConfig
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, DSRandomSampler, AudioDataLoader, \
    DSElasticDistributedSampler


class DeepSpeechDataModule(pl.LightningDataModule):

    def __init__(self,
                 labels: list,
                 data_cfg: DataConfig,
                 normalize: bool):
        super().__init__()
        self.train_path = to_absolute_path(data_cfg.train_path)
        self.val_path = to_absolute_path(data_cfg.val_path)
        self.labels = labels
        self.data_cfg = data_cfg
        self.spect_cfg = data_cfg.spect
        self.aug_cfg = data_cfg.augmentation
        self.normalize = normalize

    @property
    def is_distributed(self):
        if 'trainer' in self.__dict__:
            return self.trainer.devices > 1
        return False

    def train_dataloader(self):
        try:
            train_dataset = self._create_dataset(self.train_path)
        except Exception as e:
            print(f"Error occurred while loading training set. Error: {e}")
            return None
        if self.is_distributed:
            train_sampler = DSElasticDistributedSampler(
                dataset=train_dataset,
                batch_size=self.data_cfg.batch_size
            )
        else:
            train_sampler = DSRandomSampler(
                dataset=train_dataset,
                batch_size=self.data_cfg.batch_size
            )
        train_loader = AudioDataLoader(
            dataset=train_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_sampler=train_sampler
        )
        return train_loader

    def val_dataloader(self):
        try:
            val_dataset = self._create_dataset(self.val_path)
        except Exception as e:
            print(f"Error occurred while loading validation set. Error: {e}")
            return None
        val_loader = AudioDataLoader(
            dataset=val_dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=self.data_cfg.batch_size
        )
        return val_loader

    def _create_dataset(self, input_path):
        try:
            dataset = SpectrogramDataset(
                audio_conf=self.spect_cfg,
                input_path=input_path,
                labels=self.labels,
                normalize=True,
                aug_cfg=self.aug_cfg
            )
        except Exception as e:
            print(f"Error occurred while creating dataset. Error: {e}")
            return None
        return dataset