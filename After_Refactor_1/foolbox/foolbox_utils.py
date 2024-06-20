import os
from typing import Optional, Tuple, Any
import numpy as np
from PIL import Image
import eagerpy as ep
import warnings

from .models import Model
from .types import Bounds


def accuracy(fmodel: Model, inputs: Any, labels: Any) -> float:
    """
    Calculate the accuracy of a model's predictions.

    :param fmodel: The model to evaluate.
    :param inputs: The input data.
    :param labels: The corresponding target labels.
    :return: The accuracy as a float value.
    """
    inputs_, labels_ = ep.astensors(inputs, labels)
    del inputs, labels

    predictions = fmodel(inputs_).argmax(axis=-1)
    accuracy = (predictions == labels_).float32().mean()
    return accuracy.item()


def samples(
    fmodel: Model,
    dataset: str = "imagenet",
    index: int = 0,
    batchsize: int = 1,
    shape: Tuple[int, int] = (224, 224),
    data_format: Optional[str] = None,
    bounds: Optional[Bounds] = None,
) -> Any:
    """
    Generate samples from a dataset.

    :param fmodel: The model to generate samples for.
    :param dataset: The dataset to use.
    :param index: The starting index of the samples.
    :param batchsize: The number of samples to generate.
    :param shape: The shape of each sample image.
    :param data_format: The data format of the samples.
    :param bounds: The bounds of the samples' pixel values.
    :return: Tuple containing the generated images and labels.
    """
    if data_format is None:
        data_format = fmodel.data_format if hasattr(fmodel, "data_format") else None

    if data_format is None:
        raise ValueError("data_format could not be inferred, please specify it explicitly")

    if bounds is None:
        bounds = fmodel.bounds

    images, labels = _generate_samples(
        dataset=dataset,
        index=index,
        batchsize=batchsize,
        shape=shape,
        data_format=data_format,
        bounds=bounds,
    )

    if hasattr(fmodel, "dummy") and fmodel.dummy is not None:
        images = ep.from_numpy(fmodel.dummy, images).raw
        labels = ep.from_numpy(fmodel.dummy, labels).raw
    else:
        warnings.warn(f"unknown model type {type(fmodel)}, returning NumPy arrays")

    return images, labels


def _generate_samples(
    dataset: str,
    index: int,
    batchsize: int,
    shape: Tuple[int, int],
    data_format: str,
    bounds: Bounds,
) -> Tuple[Any, Any]:
    """
    Generate samples from a specific dataset.

    :param dataset: The dataset name.
    :param index: The starting index of the samples.
    :param batchsize: The number of samples to generate.
    :param shape: The shape of each sample image.
    :param data_format: The data format of the samples.
    :param bounds: The bounds of the samples' pixel values.
    :return: Tuple containing the generated images and labels.
    """
    images, labels = [], []
    basepath = os.path.dirname(__file__)
    samplepath = os.path.join(basepath, "data")
    files = os.listdir(samplepath)

    if batchsize > 20:
        warnings.warn("samples() has only 20 samples and repeats itself if batchsize > 20")

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

    if bounds != (0, 255):
        images_ = images_ / 255 * (bounds[1] - bounds[0]) + bounds[0]
    return images_, labels_