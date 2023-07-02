#!/usr/bin/env python3
"""
A simple example that demonstrates how to run a single attack against
a PyTorch ResNet-18 model for different epsilons and how to then report
the robust accuracy.
"""
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

def load_model():
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225],
                         axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel

def get_data(fmodel):
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    return images, labels

def calculate_clean_accuracy(fmodel, images, labels):
    clean_acc = accuracy(fmodel, images, labels)
    return clean_acc

def run_attack(fmodel, images, labels):
    attack = LinfPGD()
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=EPSILONS)
    return raw_advs, clipped_advs, success

def calculate_robust_accuracy(success):
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    return robust_accuracy

def calculate_accuracy(fmodel, advs, labels):
    accuracy = accuracy(fmodel, advs, labels)
    return accuracy

def calculate_perturbation_sizes(advs, images):
    perturbation_sizes = (advs - images).norms.linf(axis=(1, 2, 3)).numpy()
    return perturbation_sizes

def print_results(epsilons, robust_accuracy, advs, labels):
    print("robust accuracy for perturbations with")
    for eps, acc, advs_ in zip(epsilons, robust_accuracy, advs):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        print("    perturbation sizes:")
        perturbation_sizes = calculate_perturbation_sizes(advs_, images)
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc == 0:
            break

def main() -> None:
    model = load_model()
    images, labels = get_data(model)
    clean_acc = calculate_clean_accuracy(model, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    raw_advs, clipped_advs, success = run_attack(model, images, labels)
    robust_accuracy = calculate_robust_accuracy(success)
    print_results(EPSILONS, robust_accuracy, clipped_advs, labels)

if __name__ == "__main__":
    main()