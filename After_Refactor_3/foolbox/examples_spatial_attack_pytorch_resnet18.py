#!/usr/bin/env python3
from typing import Tuple

import eagerpy as ep
import foolbox.attacks as fa
from foolbox import PyTorchModel, accuracy, samples
import torchvision.models as models

# Constants for preprocessing
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
AXIS = -3

# Constants for the attack
MAX_TRANSLATION = 6
NUM_TRANSLATIONS = 6
MAX_ROTATION = 20
NUM_ROTATIONS = 5

def main() -> None:
    # Instantiate a model
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=MEAN, std=STD, axis=AXIS)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    # Get data and test the model
    images, labels = ep.astensors(*get_samples(fmodel, batchsize=16))
    clean_acc = get_accuracy(fmodel, images, labels) * 100
    print(f"clean accuracy:  {clean_acc:.1f} %")

    # Run the spatial attack
    success_rate, robust_accuracy = spatial_attack(fmodel, images, labels)
    print(f"attack success:  {success_rate:.1f} % (for the specified rotation and translation bounds)")
    print(f"robust accuracy: {robust_accuracy:.1f} % (for the specified rotation and translation bounds)")

def get_samples(fmodel: PyTorchModel, dataset: str, batchsize: int) -> Tuple[ep.Tensor, ep.Tensor]:
    """Get data samples from the model"""
    return ep.astensors(*samples(fmodel, dataset=dataset, batchsize=batchsize))

def get_accuracy(fmodel: PyTorchModel, images: ep.Tensor, labels: ep.Tensor) -> float:
    """Calculate the accuracy of the model"""
    return accuracy(fmodel, images, labels)

def spatial_attack(fmodel: PyTorchModel, images: ep.Tensor, labels: ep.Tensor) -> Tuple[float, float]:
    """Run the spatial attack and return the success rate and robust accuracy"""
    attack = fa.SpatialAttack(
        max_translation=MAX_TRANSLATION,
        num_translations=NUM_TRANSLATIONS,
        max_rotation=MAX_ROTATION,
        num_rotations=NUM_ROTATIONS
    )

    adversarial_images, _, success = attack(fmodel, images, labels)
    success_rate = success.float32().mean().item() * 100
    robust_accuracy = 100 - success_rate

    return success_rate, robust_accuracy

if __name__ == "__main__":
    main()