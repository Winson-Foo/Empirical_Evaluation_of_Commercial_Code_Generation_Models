#!/usr/bin/env python3

import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
from foolbox.attacks.base import get_criterion


class Attack(LinfPGD):
    def value_and_grad(self, loss_fn, x):
        val1 = loss_fn(x)
        loss_fn2 = self.get_loss_fn(self.model2, self.labels)
        _, grad2 = ep.value_and_grad(loss_fn2, x)
        return val1, grad2

    def run(self, model, inputs, criterion, *, epsilon, **kwargs):
        criterion_ = get_criterion(criterion)
        self.labels = criterion_.labels
        return super().run(model, inputs, criterion_, epsilon=epsilon, **kwargs)


def initialize_model() -> PyTorchModel:
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel


def get_data(model: PyTorchModel) -> tuple:
    images, labels = ep.astensors(*samples(model, dataset="imagenet", batchsize=16))
    return images, labels


def run_attack(model: PyTorchModel, images: ep.Tensor, labels: ep.Tensor, epsilons: list) -> tuple:
    attack = Attack()
    raw_advs, clipped_advs, success = attack(model, images, labels, epsilons=epsilons)
    return raw_advs, clipped_advs, success


def calculate_robust_accuracy(success: ep.Tensor) -> ep.Tensor:
    return 1 - success.float32().mean(axis=-1)


def print_robust_accuracy(epsilons: list, robust_accuracy: ep.Tensor, clipped_advs: ep.Tensor, model: PyTorchModel, labels: ep.Tensor) -> None:
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        print("    perturbation sizes:")
        perturbation_sizes = (clipped_advs - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break


def main() -> None:
    fmodel = initialize_model()

    images, labels = get_data(fmodel)
    clean_acc = accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    model2 = fmodel

    raw_advs, clipped_advs, success = run_attack(fmodel, images, labels, epsilons)

    robust_accuracy = calculate_robust_accuracy(success)
    print_robust_accuracy(epsilons, robust_accuracy, clipped_advs, fmodel, labels)


if __name__ == "__main__":
    main()