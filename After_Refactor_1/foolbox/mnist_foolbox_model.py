#!/usr/bin/env python3
import os
from typing import Tuple

import torch
import torch.nn as nn
from foolbox.models import PyTorchModel
from foolbox.utils import accuracy, samples


def load_model(model_path: str) -> nn.Module:
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
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def create_foolbox_model(model: nn.Module) -> PyTorchModel:
    preprocessing = dict(mean=0.1307, std=0.3081)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel


def test_model(fmodel: PyTorchModel, dataset: str, batchsize: int) -> float:
    images, labels = samples(fmodel, dataset=dataset, batchsize=batchsize)
    accuracy_score = accuracy(fmodel, images, labels)
    return accuracy_score


if __name__ == "__main__":
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mnist_cnn.pth")
    model = load_model(model_path)
    fmodel = create_foolbox_model(model)
    accuracy_score = test_model(fmodel, dataset="mnist", batchsize=20)
    print(accuracy_score)