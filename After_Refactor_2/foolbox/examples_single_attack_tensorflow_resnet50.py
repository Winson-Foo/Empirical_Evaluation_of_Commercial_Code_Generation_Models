#!/usr/bin/env python3

import tensorflow as tf
import eagerpy as ep
from foolbox import TensorFlowModel, accuracy, samples, Model
from foolbox.attacks import LinfPGD


def instantiate_model():
    model = tf.keras.applications.ResNet50(weights="imagenet")
    preprocessing = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel: Model = TensorFlowModel(model, bounds=(0, 255), preprocessing=preprocessing)
    fmodel = fmodel.transform_bounds((0, 1))
    return fmodel


def get_data(model):
    images, labels = ep.astensors(*samples(model, dataset="imagenet", batchsize=16))
    return images, labels


def evaluate_model(model, images, labels):
    clean_accuracy = accuracy(model, images, labels)
    print(f"Clean accuracy: {clean_accuracy * 100:.1f} %")


def apply_attack(attack, model, images, labels, epsilons):
    raw_advs, clipped_advs, success = attack(model, images, labels, epsilons=epsilons)
    return raw_advs, clipped_advs, success


def calculate_robust_accuracy(success):
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    return robust_accuracy


def evaluate_robust_accuracy(model, advs, labels):
    accuracy = accuracy(model, advs, labels)
    perturbation_sizes = (advs - images).norms.linf(axis=(1, 2, 3)).numpy()
    return accuracy, perturbation_sizes


def main() -> None:
    # Instantiate a model
    model = instantiate_model()

    # Get data and test the model
    images, labels = get_data(model)
    evaluate_model(model, images, labels)

    # Apply the attack
    attack = LinfPGD()
    epsilons = [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.003, 0.01, 0.1, 0.3, 0.5, 1.0]
    raw_advs, clipped_advs, success = apply_attack(attack, model, images, labels, epsilons)

    # Calculate and report the robust accuracy
    robust_accuracy = calculate_robust_accuracy(success)
    print("Robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")

    # Evaluate robust accuracy manually
    print("\nWe can also manually check this:")
    print("\nRobust accuracy for perturbations with")
    for eps, advs_ in zip(epsilons, clipped_advs):
        accuracy, perturbation_sizes = evaluate_robust_accuracy(model, advs_, labels)
        print(f"Linf norm ≤ {eps:<6}: {accuracy * 100:4.1f} %")
        print("    Perturbation sizes:")
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if accuracy == 0:
            break


if __name__ == "__main__":
    main()