#!/usr/bin/env python3
# mypy: no-disallow-untyped-defs
"""
Sometimes one wants to replace the gradient of a model with a different gradient
from another model to make the attack more reliable. That is, the forward pass
should go through model 1, but the backward pass should go through model 2.
This example shows how that can be done in Foolbox.
"""
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD
from foolbox.attacks.base import get_criterion

def load_pretrained_model() -> PyTorchModel:
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    return PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)


def get_data() -> tuple:
    model = load_pretrained_model()
    images, labels = ep.astensors(*samples(model, dataset="imagenet", batchsize=16))
    return (model, images, labels)


def replace_gradient(model, inputs, criterion, model2):
    class Attack(LinfPGD):
        def value_and_grad(self, loss_fn, x):
            val1 = loss_fn(x)
            loss_fn2 = self.get_loss_fn(model2, self.labels)
            _, grad2 = ep.value_and_grad(loss_fn2, x)
            return val1, grad2

        def run(self, model, inputs, criterion, *, epsilon, **kwargs):
            criterion_ = get_criterion(criterion)
            self.labels = criterion_.labels
            return super().run(model, inputs, criterion_, epsilon=epsilon, **kwargs)

    attack = Attack()
    epsilons = [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.1, 0.3, 0.5, 1.0]
    raw_advs, clipped_advs, success = attack(model, inputs, criterion, epsilons=epsilons)

    return raw_advs, clipped_advs, success


def evaluate_robust_accuracy(model, adversarials, labels, epsilons):
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


def evaluate_manually(model, adversarials, labels, epsilons):
    print()
    print("we can also manually check this:")
    print()
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(epsilons, adversarials):
        acc2 = accuracy(model, advs_, labels)
        print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
        print("    perturbation sizes:")
        perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break


def main() -> None:
    model, images, labels = get_data()
    clean_acc = accuracy(model, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    model2 = model

    raw_advs, clipped_advs, success = replace_gradient(model, images, labels, model2)

    evaluate_robust_accuracy(model, clipped_advs, labels, epsilons)

    evaluate_manually(model, clipped_advs, labels, epsilons)


if __name__ == "__main__":
    main()