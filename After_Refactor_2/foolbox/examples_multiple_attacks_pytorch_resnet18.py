#!/usr/bin/env python3
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
import numpy as np

def load_pretrained_model():
    # Load and return a pretrained ResNet-18 model
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel

def get_test_data(model, batchsize=16):
    # Generate test images and labels using the given model
    images, labels = ep.astensors(*samples(model, dataset="imagenet", batchsize=batchsize))
    return images, labels

def evaluate_clean_accuracy(model, images, labels):
    # Evaluate the clean accuracy of the given model on the test images
    clean_acc = accuracy(model, images, labels)
    print(f"clean accuracy: {clean_acc * 100:.1f} %\n")
    return clean_acc

def run_attacks(model, images, labels, attacks, epsilons):
    # Run given attacks on the model with different epsilon values and calculate success rates
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
    # Calculate the robust accuracy of the model using the best attack per sample
    robust_accuracy = 1.0 - attack_success.max(axis=0).mean(axis=-1)
    print("\n" + "-" * 79 + "\n")
    print("worst case (best attack per-sample)")
    print("  ", robust_accuracy.round(2), "\n")
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

def main():
    # Main function to run the code
    model = load_pretrained_model()
    images, labels = get_test_data(model)
    clean_acc = evaluate_clean_accuracy(model, images, labels)

    attacks = [
        fa.FGSM(),
        fa.LinfPGD(),
        fa.LinfBasicIterativeAttack(),
        fa.LinfAdditiveUniformNoiseAttack(),
        fa.LinfDeepFoolAttack(),
    ]

    epsilons = [
        0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.1, 0.3, 0.5, 1.0,
    ]

    print("epsilons")
    print(epsilons, "\n")

    attack_success = run_attacks(model, images, labels, attacks, epsilons)
    calculate_robust_accuracy(attack_success)

if __name__ == "__main__":
    main()