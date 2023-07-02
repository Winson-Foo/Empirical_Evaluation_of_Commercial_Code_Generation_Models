#!/usr/bin/env python3

from typing import Tuple, List

import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD

EPSILONS = [
    0.0,
    0.0002,
    0.0005,
    0.0008,
    0.001,
    0.0015,
    0.002,
    0.003,
    0.01,
    0.1,
    0.3,
    0.5,
    1.0,
]


def load_model() -> PyTorchModel:
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    return PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)


def evaluate_clean_accuracy(model: PyTorchModel, images: ep.Tensor, labels: ep.Tensor) -> float:
    clean_acc = accuracy(model, images, labels)
    return clean_acc * 100


def run_attack(model: PyTorchModel, images: ep.Tensor, labels: ep.Tensor) -> Tuple[List[ep.Tensor], List[ep.Tensor]]:
    attack = LinfPGD()
    raw_advs, clipped_advs, _ = attack(model, images, labels, epsilons=EPSILONS)
    return raw_advs, clipped_advs


def calculate_robust_accuracy(labels: ep.Tensor, success: ep.Tensor) -> List[float]:
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    return [acc.item() * 100 for acc in robust_accuracy]


def evaluate_robustness(model: PyTorchModel, advs: List[ep.Tensor], labels: ep.Tensor) -> None:
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(EPSILONS, advs):
        acc = accuracy(model, advs_, labels)
        print(f"  Linf norm ≤ {eps:<6}: {acc * 100:4.1f} %")
        print("    perturbation sizes:")
        perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc == 0:
            break


def main() -> None:
    model = load_model()

    images, labels = ep.astensors(*samples(model, dataset="imagenet", batchsize=16))

    clean_acc = evaluate_clean_accuracy(model, images, labels)
    print(f"clean accuracy: {clean_acc:.1f} %")

    raw_advs, clipped_advs = run_attack(model, images, labels)

    robust_accuracy = calculate_robust_accuracy(labels, success)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(EPSILONS, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc:4.1f} %")

    print("\nwe can also manually check this:\n")
    evaluate_robustness(model, clipped_advs, labels)


if __name__ == "__main__":
    main()