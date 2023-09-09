import os
from keras.models import model_from_json

def save_model(name_prefix, model):
    """
    Save a Keras sequential model into files.

    Given a Keras sequential model, save the model with the given file path prefix.
    It saves the model into a JSON file, and an HDF5 file (.h5).

    :param name_prefix: Prefix of the paths of the model files
    :param model: Keras sequential model to be saved
    :return: None
    """
    try:
        # Save model architecture
        model_json = model.to_json()
        with open(f'{name_prefix}.json', 'w') as json_file:
            json_file.write(model_json)

        # Save model weights
        model.save_weights(f'{name_prefix}.h5')
    except Exception as e:
        print(f"Error occurred while saving model: {e}")


def load_model(name_prefix):
    """
    Load a Keras sequential model from files.

    Given the prefix of the file paths, load a Keras sequential model
    from a JSON file and an HDF5 file.

    :param name_prefix: Prefix of the paths of the model files
    :return: Keras sequential model
    """
    try:
        # Load model architecture
        if not os.path.exists(f'{name_prefix}.json'):
            raise ValueError(f"The file {name_prefix}.json does not exist.")
        with open(f'{name_prefix}.json', 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)

        # Load model weights
        if not os.path.exists(f'{name_prefix}.h5'):
            raise ValueError(f"The file {name_prefix}.h5 does not exist.")
        model.load_weights(f'{name_prefix}.h5')
    except Exception as e:
        print(f"Error occurred while loading model: {e}")
        model = None

    return model