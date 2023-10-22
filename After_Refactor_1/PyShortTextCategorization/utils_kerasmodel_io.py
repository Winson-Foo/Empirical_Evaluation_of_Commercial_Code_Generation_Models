from typing import Tuple
import keras.models as models


MODEL_JSON_PATH = "model.json"
MODEL_WEIGHTS_PATH = "model.h5"


def save_model(model: models.Model, filepath_prefix: str) -> None:
    """Save a Keras sequential model into files.

    Given a Keras sequential model, save the model with the given file path prefix.
    It saves the model into a JSON file, and an HDF5 file (.h5).

    :param model: Keras sequential model to be saved.
    :param filepath_prefix: Prefix of the paths of the model files.
    :return: None.
    """
    model_json = model.to_json()
    with open(filepath_prefix + ".json", "w") as f:
        f.write(model_json)
    model.save_weights(filepath_prefix + ".h5")


def load_model(filepath_prefix: str) -> models.Model:
    """Load a Keras sequential model from files.

    Given the prefix of the file paths, load a Keras sequential model from
    a JSON file and an HDF5 file.

    :param filepath_prefix: Prefix of the paths of the model files.
    :return: Keras sequential model.
    """
    with open(filepath_prefix + ".json", "r") as f:
        model_json = f.read()
    model = models.model_from_json(model_json)
    model.load_weights(filepath_prefix + ".h5")
    return model