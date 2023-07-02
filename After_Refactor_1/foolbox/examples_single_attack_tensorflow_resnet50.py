#!/usr/bin/env python3
import tensorflow as tf
import eagerpy as ep
from foolbox import TensorFlowModel, accuracy, samples, Model
from foolbox.attacks import LinfPGD


def instantiate_model() -> Model:
    model = tf.keras.applications.ResNet50(weights="imagenet")
    pre = dict(flip_axis=-1, mean=[104.0, 116.0, 123.0])  # RGB to BGR
    fmodel: Model = TensorFlowModel(model, bounds=(0, 255), preprocessing=pre)
    fmodel = fmodel.transform_bounds((0, 1))
    return fmodel


def get_data(fmodel: Model):
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    return images, labels


def print_clean_accuracy(accuracy: float):
    print(f"clean accuracy:  {accuracy * 100:.1f} %")


def apply_attack(fmodel: Model, images, labels):
    attack = LinfPGD()
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
    raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=epsilons)
    return raw_advs, clipped_advs, success


def print_robust_accuracy(epsilons, robust_accuracy):
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


def manually_check_robust_accuracy(fmodel: Model, epsilons, clipped_advs, labels):
    print()
    print("we can also manually check this:")
    print()
    print("robust accuracy for perturbations with")
    for eps, advs_ in zip(epsilons, clipped_advs):
        acc2 = accuracy(fmodel, advs_, labels)
        print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
        print("    perturbation sizes:")
        perturbation_sizes = (advs_ - images).norms.linf(axis=(1, 2, 3)).numpy()
        print("    ", str(perturbation_sizes).replace("\n", "\n" + "    "))
        if acc2 == 0:
            break


def main() -> None:
    fmodel = instantiate_model()
    images, labels = get_data(fmodel)
    clean_acc = accuracy(fmodel, images, labels)
    print_clean_accuracy(clean_acc)

    raw_advs, clipped_advs, success = apply_attack(fmodel, images, labels)

    robust_accuracy = 1 - success.float32().mean(axis=-1)
    print_robust_accuracy(epsilons, robust_accuracy)

    manually_check_robust_accuracy(fmodel, epsilons, clipped_advs, labels)


if __name__ == "__main__":
    main()