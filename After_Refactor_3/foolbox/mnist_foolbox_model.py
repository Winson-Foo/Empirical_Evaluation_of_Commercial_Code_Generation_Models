#!/usr/bin/env python3
import os

import torch
import torch.nn as nn
from foolbox.models import PyTorchModel
from foolbox.utils import accuracy, samples

MNIST_MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mnist_cnn.pth")
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def create() -> PyTorchModel:
    """
    Create and load a PyTorch model for MNIST classification.
    
    Returns:
        fmodel (PyTorchModel): The loaded PyTorch model.
    """
    model = create_model()
    load_model(model, MNIST_MODEL_PATH)
    model.eval()
    preprocessing = dict(mean=MNIST_MEAN, std=MNIST_STD)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel


def create_model() -> nn.Sequential:
    """
    Create the PyTorch model for MNIST classification.
    
    Returns:
        model (nn.Sequential): The created PyTorch model.
    """
    return nn.Sequential(
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


def load_model(model: nn.Sequential, path: str) -> None:
    """
    Load the trained model weights from the given path.
    
    Args:
        model (nn.Sequential): The model to load the weights into.
        path (str): The path to the trained model weights.
    """
    model.load_state_dict(torch.load(path))  # type: ignore


if __name__ == "__main__":
    # test the model
    fmodel = create()
    images, labels = samples(fmodel, dataset="mnist", batchsize=20)
    print(accuracy(fmodel, images, labels))