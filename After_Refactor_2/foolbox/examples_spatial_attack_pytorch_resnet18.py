import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa


def instantiate_model():
    # instantiate a model (could also be a TensorFlow or JAX model)
    model = models.resnet18(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    return fmodel


def get_data(fmodel):
    # get data and test the model
    # wrapping the tensors with ep.astensors is optional, but it allows
    # us to work with EagerPy tensors in the following
    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=16))
    return images, labels


def evaluate_model(fmodel, images, labels):
    clean_acc = accuracy(fmodel, images, labels) * 100
    print(f"clean accuracy:  {clean_acc:.1f} %")


def perform_spatial_attack(images, labels):
    # the attack trys a combination of specified rotations and translations to an image
    # stops early if adversarial shifts and translations for all images are found
    attack = fa.SpatialAttack(
        max_translation=6,  # 6px so x in [x-6, x+6] and y in [y-6, y+6]
        num_translations=6,  # number of translations in x, y.
        max_rotation=20,  # +- rotation in degrees
        num_rotations=5,  # number of rotations
        # max total iterations = num_rotations * num_translations**2
    )

    xp_, _, success = attack(fmodel, images, labels)
    suc = success.float32().mean().item() * 100
    print(
        f"attack success:  {suc:.1f} %"
        " (for the specified rotation and translation bounds)"
    )
    print(
        f"robust accuracy: {100 - suc:.1f} %"
        " (for the specified rotation and translation bounds)"
    )


def main() -> None:
    fmodel = instantiate_model()
    images, labels = get_data(fmodel)
    evaluate_model(fmodel, images, labels)
    perform_spatial_attack(images, labels)


if __name__ == "__main__":
    main()