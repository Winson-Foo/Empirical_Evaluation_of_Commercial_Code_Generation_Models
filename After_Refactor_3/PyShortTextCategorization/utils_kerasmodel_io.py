from keras.models import model_from_json
import os


def save_model(model, name_prefix):
    """Save a Keras sequential model to files.

    Given a Keras sequential model, save the model with the given file path prefix.
    It saves the model into a JSON file, and an HDF5 file (.h5).

    Args:
        model (keras.models.Model): Keras sequential model to be saved.
        name_prefix (str): Prefix of the paths of the model files.

    Returns:
        None
    """
    model_json = model.to_json()
    with open(name_prefix+'.json', 'w') as file:
        file.write(model_json)
    model.save_weights(name_prefix+'.h5')


def load_model(name_prefix):
    """Load a Keras sequential model from files.

    Given the prefix of the file paths, load a Keras sequential model from
    a JSON file and an HDF5 file.

    Args:
        name_prefix (str): Prefix of the paths of the model files.

    Returns:
        keras.models.Model: Keras sequential model.
    """
    model_file = name_prefix + '.json'
    weights_file = name_prefix + '.h5'

    if not os.path.exists(model_file) or not os.path.exists(weights_file):
        raise IOError('Model files do not exist')

    with open(model_file, 'r') as file:
        model_json = file.read()
    model = model_from_json(model_json)

    try:
        model.load_weights(weights_file)
    except ValueError:
        raise ValueError('Cannot load weights of the model')

    return model