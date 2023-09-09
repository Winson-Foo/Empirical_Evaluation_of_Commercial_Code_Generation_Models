from time import time
import imageio
import yaml
import numpy as np
from pathlib import Path
from ISR.utils.logger import get_logger
from ISR.utils.utils import get_timestamp

# Constants
EXTENSIONS = ('.jpeg', '.jpg', '.png')  # file extensions that are admitted


class Predictor:
    """The predictor class handles prediction, given an input model.

    Loads the images in the input directory, executes training given a model
    and saves the results in the output directory.
    Can receive a path for the weights or can let the user browse through the
    weights directory for the desired weights.

    Args:
        input_dir: string, path to the input directory.
        output_dir: string, path to the output directory.
        verbose: bool.

    Attributes:
        img_ls: list of image files in input_dir.
    """

    def __init__(self, input_dir, output_dir='./data/output', verbose=True):
        self.input_dir = Path(input_dir)
        self.img_ls = [f for f in self.input_dir.iterdir() if f.suffix in EXTENSIONS]
        if len(self.img_ls) < 1:
            raise ValueError('No valid image files found (check config file).')
        self.output_dir = Path(output_dir) / self.input_dir.name
        self.logger = get_logger(__name__)
        if not verbose:
            self.logger.setLevel(40)
        self._create_output_dir()

    def _create_output_dir(self):
        """ Creates the output directory if it does not exist. """
        if not self.output_dir.exists():
            self.logger.info('Creating output directory:\n{}'.format(self.output_dir))
            self.output_dir.mkdir(parents=True)

    def get_predictions(self, model, weights_path):
        """ Runs the prediction on all images in the input directory. """

        self._validate_weights_path(weights_path)
        weights_conf = self._load_weights()
        out_folder = self._create_results_folder()
        if weights_conf:
            self._save_weights_config(weights_conf, out_folder)
        self._predict_and_store_images(model, out_folder)

    def _validate_weights_path(self, weights_path):
        """ Validates the weights path and raises an error if it is invalid. """
        if weights_path is None:
            raise ValueError('Weights path not specified (check config file).')
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise ValueError('Invalid weights path: {}'.format(weights_path))

    def _load_weights(self):
        """ Loads weights from the specified path and returns the weights config. """
        self.model.model.load_weights(str(self.weights_path))
        session_config_path = self.weights_path.parent / 'session_config.yml'
        if session_config_path.exists():
            conf = yaml.load(session_config_path.read_text(), Loader=yaml.FullLoader)
        else:
            self.logger.warning('Could not find weights training configuration')
            conf = {}
        conf.update({'pre-trained-weights': self.weights_path.name})
        return conf

    def _create_results_folder(self):
        """ Creates the output folder for the prediction results. """
        out_folder = self.output_dir / self._make_basename() / get_timestamp()
        self.logger.info('Results in:\n > {}'.format(out_folder))
        if out_folder.exists():
            self.logger.warning('Directory exists, might overwrite files')
        else:
            out_folder.mkdir(parents=True)
        return out_folder

    def _make_basename(self):
        """ Combines the name of the model and its architecture's parameters. """
        params = [self.model.name]
        for param in np.sort(list(self.model.params.keys())):
            params.append('{g}{p}'.format(g=param, p=self.model.params[param]))
        return '-'.join(params)

    def _save_weights_config(self, weights_conf, out_folder):
        """ Saves the weights configuration to a yaml file. """
        yaml.dump(weights_conf, (out_folder / 'weights_config.yml').open('w'))

    def _predict_and_store_images(self, model, out_folder):
        """ Runs the prediction on all images and stores the results in the output directory. """
        for img_path in self.img_ls:
            output_path = out_folder / img_path.name
            self.logger.info('Processing file\n > {}'.format(img_path))
            start = time()
            sr_img = self._forward_pass(model, img_path)
            self._store_image(sr_img, output_path)
            end = time()
            self.logger.info('Elapsed time: {}s'.format(end - start))

    def _forward_pass(self, model, file_path):
        """ Runs the forward pass through the model and returns the super-resolution image. """
        lr_img = imageio.imread(file_path)
        if lr_img.shape[2] == 3:
            sr_img = model.predict(lr_img)
            return sr_img
        else:
            self.logger.error('{} is not an image with 3 channels.'.format(file_path))

    def _store_image(self, sr_img, output_path):
        """ Stores the super-resolution image to the specified output path. """
        self.logger.info('Result in: {}'.format(output_path))
        imageio.imwrite(output_path, sr_img)