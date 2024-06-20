#!/usr/bin/env python3

import tensorflow as tf
import eagerpy as ep
from foolbox import TensorFlowModel, accuracy, samples, Model
from foolbox.attacks import LinfPGD


def main() -> None:
    """
    Main function to run the code.
    """
    model = load_resnet_model()
    fmodel = wrap_model_with_foolbox(model)
    images, labels = get_data(fmodel)
    clean_acc = calculate_clean_accuracy(fmodel, images, labels)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")

    epsilons = get_epsilon_values()
    raw_advs, clipped_advs, success = perform_attack(fmodel, images, labels, epsilons)
    report_robust_accuracy(epsilons, success)

    print()
    print("we can also manually check this:")
    print()
    manual_check(epsilons, clipped_advs, labels, fmodel)


def load_resnet_model() -> tf.keras.Model:
    """
    Load and return the ResNet50 model.
    """
    return tf.keras.applications.ResNet50(weights="imagenet")


def wrap_model_with_foolbox(model: tf.keras.Model) -> Model:
    """
    Wrap the TensorFlow model with Foolbox Model.
    """
    preprocessing = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])
    fmodel = TensorFlowModel(model, bounds=(0, 255), preprocessing=preprocessing)
    fmodel = fmodel.transform_bounds((0, 1))
    return fmodel


def get_data(fmodel: Model) -> tuple:
    """
    Get images and labels using the Foolbox samples function.
    """
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    return images, labels


def calculate_clean_accuracy(fmodel: Model, images: ep.Tensor, labels: ep.Tensor) -> float:
    """
    Calculate the clean accuracy of the model.
    """
    return accuracy(fmodel, images, labels)


def get_epsilon_values() -> list:
    """
    Return the list of epsilon values for the attack.
    """
    return [
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


def perform_attack(fmodel: Model, images: ep.Tensor, labels: ep.Tensor, epsilons: list) -> tuple:
    """
    Perform the LinfPGD attack using Foolbox.
    """
    attack = LinfPGD()
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
    return raw_advs, clipped_advs, success


def report_robust_accuracy(epsilons: list, success: ep.Tensor) -> None:
    """
    Calculate and report the robust accuracy for different epsilon values.
    """
    robust_accuracy = 1 - success.float32().mean(axis=-1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


def manual_check(epsilons: list, clipped_advs: ep.Tensor, labels: ep.Tensor, fmodel: Model) -> None:
    """
    Manually check the robust accuracy using the clipped adversarial examples.
    Print perturbation sizes and break the loop if accuracy is 0.
    """
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(epsilons, clipped_advs):
        acc2 = accuracy(fmodel, advs_, labels)
        print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
        print("    perturbation sizes:")
        perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break


if __name__ == "__main__":
    main()