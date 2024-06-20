from typing import Optional, Tuple, Any
import eagerpy as ep
import warnings
import os
import numpy as np
from PIL import Image

from .types import Bounds
from .models import Model


class DataManager:
    def __init__(self, fmodel: Model):
        self.fmodel = fmodel

    def get_samples(
        self,
        dataset: str = "imagenet",
        index: int = 0,
        batchsize: int = 1,
        shape: Tuple[int, int] = (224, 224),
        data_format: Optional[str] = None,
        bounds: Optional[Bounds] = None,
    ) -> Any:
        self._check_data_format(data_format)
        self._check_bounds(bounds)

        images, labels = self._load_samples(
            dataset=dataset,
            index=index,
            batchsize=batchsize,
            shape=shape,
            data_format=data_format,
        )

        if hasattr(self.fmodel, "dummy") and self.fmodel.dummy is not None:  # type: ignore
            images = ep.from_numpy(self.fmodel.dummy, images).raw  # type: ignore
            labels = ep.from_numpy(self.fmodel.dummy, labels).raw  # type: ignore
        else:
            warnings.warn(f"unknown model type {type(self.fmodel)}, returning NumPy arrays")

        return images, labels

    def _check_data_format(self, data_format: Optional[str]) -> None:
        if hasattr(self.fmodel, "data_format"):
            if data_format is None:
                data_format = self.fmodel.data_format  # type: ignore
            elif data_format != self.fmodel.data_format:  # type: ignore
                raise ValueError(
                    f"data_format ({data_format}) does not match model.data_format ({self.fmodel.data_format})"  # type: ignore
                )
        elif data_format is None:
            raise ValueError(
                "data_format could not be inferred, please specify it explicitly"
            )

    def _check_bounds(self, bounds: Optional[Bounds]) -> None:
        if bounds is None:
            bounds = self.fmodel.bounds

    def _load_samples(
        self,
        dataset: str,
        index: int,
        batchsize: int,
        shape: Tuple[int, int],
        data_format: str,
    ) -> Tuple[Any, Any]:
        images, labels = [], []
        basepath = os.path.dirname(__file__)
        samplepath = os.path.join(basepath, "data")
        files = os.listdir(samplepath)

        if batchsize > 20:
            warnings.warn(
                "get_samples() has only 20 samples and repeats itself if batchsize > 20"
            )

        for idx in range(index, index + batchsize):
            i = idx % 20

            # get filename and label
            file = [n for n in files if f"{dataset}_{i:02d}_" in n][0]
            label = int(file.split(".")[0].split("_")[-1])

            # open file
            path = os.path.join(samplepath, file)
            image = Image.open(path)

            if dataset == "imagenet":
                image = image.resize(shape)

            image = np.asarray(image, dtype=np.float32)

            if image.ndim == 2:
                image = image[..., np.newaxis]

            assert image.ndim == 3

            if data_format == "channels_first":
                image = np.transpose(image, (2, 0, 1))

            images.append(image)
            labels.append(label)

        images_ = np.stack(images)
        labels_ = np.array(labels).astype(np.int64)

        return images_, labels_