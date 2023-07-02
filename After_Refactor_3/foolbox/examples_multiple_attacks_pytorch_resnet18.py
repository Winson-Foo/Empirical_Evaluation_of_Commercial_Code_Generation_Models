#!/usr/bin/env python3
from torchvision.models import resnet18
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np


def instantiate_model():
    """
    Instantiate the ResNet18 model with pretrained weights.
    """
    model = resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel


def get_data(model):
    """
    Get image and label data for testing the model.
    """
    images, labels = ep.astensors(*samples(model, dataset="imagenet", batchsize=16))
    return images, labels


def test_clean_accuracy(model, images, labels):
    """
    Test the clean accuracy of the model on the given images and labels.
    """
    clean_acc = accuracy(model, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    print("")


def run_attacks(attacks, epsilons, model, images, labels):
    """
    Run different attacks on the model with varying epsilon values.
    """
    attack_success = np.zeros((len(attacks), len(epsilons), len(images)), dtype=np.bool)
    for i, attack in enumerate(attacks):
        _, _, success = attack(model, images, labels, epsilons=epsilons)
        assert success.shape == (len(epsilons), len(images))
        success_ = success.numpy()
        assert success_.dtype == np.bool
        attack_success[i] = success_
        print(attack)
        print("  ", 1.0 - success_.mean(axis=-1).round(2))
    return attack_success


def calculate_robust_accuracy(attack_success):
    """
    Calculate and report the robust accuracy of the model.
    """
    robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
    print("")
    print("-" * 79)
    print("")
    print("worst case (best attack per-sample)")
    print("  ", robust_accuracy.round(2))
    print("")


def report_robust_accuracy(epsilons, robust_accuracy):
    """
    Report the robust accuracy for different epsilon values.
    """
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


if __name__ == "__main__":
    model = instantiate_model()
    images, labels = get_data(model)
    test_clean_accuracy(model, images, labels)

    attacks = [
        fa.FGSM(),
        fa.LinfPGD(),
        fa.LinfBasicIterativeAttack(),
        fa.LinfAdditiveUniformNoiseAttack(),
        fa.LinfDeepFoolAttack(),
    ]

    epsilons = [
        0.0,
        0.0005,
        0.001,
        0.0015,
        0.002,
        0.003,
        0.005,
        0.01,
        0.02,
        0.03,
        0.1,
        0.3,
        0.5,
        1.0,
    ]
    print("epsilons")
    print(epsilons)
    print("")

    attack_success = run_attacks(attacks, epsilons, model, images, labels)
    calculate_robust_accuracy(attack_success)
    report_robust_accuracy(epsilons, robust_accuracy)