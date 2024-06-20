#!/usr/bin/env python3

"""
The spatial attack is a very special attack because it tries to find adversarial
perturbations using a set of translations and rotations rather then in an Lp ball.
It therefore has a slightly different interface.
"""

import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa


def instantiate_model() -> PyTorchModel:
    """
    Instantiate a PyTorch model and return a preprocessed model.
    """
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        axis=-3
    )
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel


def get_data(fmodel: PyTorchModel) -> tuple:
    """
    Get data and labels for testing the model.
    """
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    return images, labels


def calculate_clean_accuracy(fmodel: PyTorchModel, images: ep.Tensor, labels: ep.Tensor) -> float:
    """
    Calculate the clean accuracy of the model.
    """
    clean_acc = accuracy(fmodel, images, labels) * 100
    return clean_acc


def run_spatial_attack(fmodel: PyTorchModel, images: ep.Tensor, labels: ep.Tensor) -> tuple:
    """
    Run spatial attack on the model and return the success rate and robust accuracy.
    """
    attack = fa.SpatialAttack(
        max_translation=6,
        num_translations=6,
        max_rotation=20,
        num_rotations=5
    )
    xp_, _, success = attack(fmodel, images, labels)
    suc = success.float32().mean().item() * 100
    return suc


def main() -> None:
    fmodel = instantiate_model()
    images, labels = get_data(fmodel)
    clean_acc = calculate_clean_accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc:.1f} %")

    suc = run_spatial_attack(fmodel, images, labels)
    print(f"attack success:  {suc:.1f} % (for the specified rotation and translation bounds)")
    print(f"robust accuracy: {100 - suc:.1f} % (for the specified rotation and translation bounds)")


if __name__ == "__main__":
    main()