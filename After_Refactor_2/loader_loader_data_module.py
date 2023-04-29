import pytorch_lightning as pl
from hydra.utils import to_absolute_path

from deepspeech_pytorch.configs.train_config import DataConfig
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, DSRandomSampler, AudioDataLoader, \
    DSElasticDistributedSampler


class DeepSpeechDataModule(pl.LightningDataModule):

    def __init__(self, labels: list, data_config: DataConfig, normalize: bool):
        super().__init__()
        self.train_path = to_absolute_path(data_config.train_path)
        self.val_path = to_absolute_path(data_config.val_path)
        self.labels = labels
        self.data_config = data_config
        self.spectrogram_config = data_config.spect
        self.augmentation_config = data_config.augmentation
        self.normalize = normalize
        self.num_workers = data_config.num_workers
        self.batch_size = data_config.batch_size

    @property
    def is_distributed(self) -> bool:
        return self.trainer.devices > 1

    def train_dataloader(self) -> AudioDataLoader:
        train_dataset = self._create_dataset(self.train_path)
        if self.is_distributed:
            train_sampler = DSElasticDistributedSampler(
                dataset=train_dataset,
                batch_size=self.batch_size
            )
        else:
            train_sampler = DSRandomSampler(
                dataset=train_dataset,
                batch_size=self.batch_size
            )
        train_loader = AudioDataLoader(
            dataset=train_dataset,
            num_workers=self.num_workers,
            batch_sampler=train_sampler
        )
        return train_loader

    def val_dataloader(self) -> AudioDataLoader:
        val_dataset = self._create_dataset(self.val_path)
        val_loader = AudioDataLoader(
            dataset=val_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size
        )
        return val_loader

    def _create_dataset(self, input_path: str) -> SpectrogramDataset:
        dataset = SpectrogramDataset(
            audio_conf=self.spectrogram_config,
            input_path=input_path,
            labels=self.labels,
            normalize=self.normalize,
            aug_cfg=self.augmentation_config
        )
        return dataset