import pytorch_lightning as pl
from hydra.utils import to_absolute_path
from typing import List

from deepspeech_pytorch.configs.train_config import DataConfig
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, DSRandomSampler, AudioDataLoader, DSElasticDistributedSampler


class DeepSpeechDataModule(pl.LightningDataModule):
    def __init__(self, labels: List[str], data_config: DataConfig, normalize: bool):
        super().__init__()
        self.train_path = to_absolute_path(data_config.train_path)
        self.val_path = to_absolute_path(data_config.val_path)
        self.labels = labels
        self.data_config = data_config
        self.spect_cfg = data_config.spect
        self.aug_cfg = data_config.augmentation
        self.normalize = normalize
        
    def train_dataloader(self) -> AudioDataLoader:
        train_dataset = self._create_dataset(self.train_path)
        train_sampler = self._create_sampler(train_dataset)
        train_loader = self._create_loader(train_dataset, train_sampler)
        return train_loader

    def val_dataloader(self) -> AudioDataLoader:
        val_dataset = self._create_dataset(self.val_path)
        val_loader = self._create_loader(val_dataset)
        return val_loader
    
    def _create_dataset(self, input_path: str) -> SpectrogramDataset:
        dataset = SpectrogramDataset(
            audio_conf=self.spect_cfg,
            input_path=input_path,
            labels=self.labels,
            normalize=self.normalize,
            aug_cfg=self.aug_cfg
        )
        return dataset

    def _create_sampler(self, dataset: SpectrogramDataset) -> DSRandomSampler:
        if self.is_distributed():
            return DSElasticDistributedSampler(
                dataset=dataset,
                batch_size=self.data_config.batch_size
            )
        else:
            return DSRandomSampler(
                dataset=dataset,
                batch_size=self.data_config.batch_size
            )

    def _create_loader(self, dataset: SpectrogramDataset, sampler=None) -> AudioDataLoader:
        loader = AudioDataLoader(
            dataset=dataset,
            num_workers=self.data_config.num_workers,
            batch_size=self.data_config.batch_size,
            sampler=sampler
        )
        return loader
    
    def is_distributed(self) -> bool:
        return self.trainer.devices > 1