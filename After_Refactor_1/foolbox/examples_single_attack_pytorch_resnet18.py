#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
import torchvision.models as models


class Config:
    model_preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    epsilons = [
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


def instantiate_model() -> PyTorchModel:
    model = models.resnet18(pretrained=True).eval()
    return PyTorchModel(model, bounds=(0, 1), preprocessing=Config.model_preprocessing)


def evaluate_clean_accuracy(model: PyTorchModel) -> float:
    images, labels = ep.astensors(*samples(model, dataset="imagenet", batchsize=16))
    clean_acc = accuracy(model, images, labels)
    return clean_acc.item()


def perform_attack(model: PyTorchModel, epsilons: list[float]) -> tuple:
    attack = LinfPGD()
    images, labels = ep.astensors(*samples(model, dataset="imagenet", batchsize=16))
    raw_advs, clipped_advs, success = attack(model, images, labels, epsilons=epsilons)
    return raw_advs, clipped_advs, success


def calculate_robust_accuracy(success: ep.Tensor) -> ep.Tensor:
    return 1 - success.float32().mean(axis=-1)


def print_robust_accuracy(epsilons: list[float], robust_accuracy: ep.Tensor):
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


def evaluate_adversarial_accuracy(model: PyTorchModel, advs: ep.Tensor, labels: ep.Tensor) -> float:
    return accuracy(model, advs, labels)


def print_perturbation_sizes(advs: ep.Tensor, images: ep.Tensor):
    for i, adv in enumerate(advs):
        perturbation_sizes = (adv - images[i]).norms.linf(axis=(1, 2, 3)).numpy()
        print("    perturbation sizes:")
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))


def main() -> None:
    model = instantiate_model()

    clean_acc = evaluate_clean_accuracy(model)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    raw_advs, clipped_advs, success = perform_attack(model, Config.epsilons)

    robust_accuracy = calculate_robust_accuracy(success)
    print_robust_accuracy(Config.epsilons, robust_accuracy)

    print()
    print("we can also manually check this:")
    print()
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(Config.epsilons, clipped_advs):
        acc2 = evaluate_adversarial_accuracy(model, advs_, labels)
        print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
        print_perturbation_sizes(advs_, images)
        if acc2 == 0:
            break


if __name__ == "__main__":
    main()