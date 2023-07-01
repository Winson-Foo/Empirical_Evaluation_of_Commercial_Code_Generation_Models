#!/usr/bin/env python3
from typing import Tuple, List
import torch
import torch.nn as nn
from foolbox.models import PyTorchModel
from foolbox.utils import accuracy, samples


def create_model() -> PyTorchModel:
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.Linear(128, 10),
    )
    model_path = get_model_path()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preprocessing = dict(mean=0.1307, std=0.3081)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel


def get_model_path() -> str:
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_filename = "mnist_cnn.pth"
    return os.path.join(current_dir, model_filename)


def test_model() -> None:
    fmodel = create_model()
    images, labels = get_samples(fmodel, dataset="mnist", batch_size=20)
    print(calculate_accuracy(fmodel, images, labels))


def get_samples(model: PyTorchModel, dataset: str, batch_size: int) -> Tuple[List, List]:
    return samples(model, dataset=dataset, batchsize=batch_size)


def calculate_accuracy(model: PyTorchModel, images: List, labels: List) -> float:
    return accuracy(model, images, labels)


if __name__ == "__main__":
    test_model()