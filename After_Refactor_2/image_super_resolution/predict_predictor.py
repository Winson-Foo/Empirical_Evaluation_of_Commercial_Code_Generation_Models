from time import time
import yaml
import numpy as np
from pathlib import Path
import imageio
from ISR.utils.logger import get_logger
from ISR.utils.utils import get_timestamp

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
        extensions: list of accepted image extensions.
        img_ls: list of image files in input_dir.

    Methods:
        get_predictions: given a model and a string containing the weights' path,
            runs the predictions on the images contained in the input directory and
            stores the results in the output directory.
    """

    def __init__(self, input_dir, output_dir='./data/output', verbose=True):
        self.input_dir = Path(input_dir)
        self.data_name = self.input_dir.name
        self.output_dir = Path(output_dir) / self.data_name
        self.logger = get_logger(__name__)
        if not verbose:
            self.logger.setLevel(40)
        self.extensions = ('.jpeg', '.jpg', '.png') 
        self.img_ls = [f for f in self.input_dir.iterdir() if f.suffix in self.extensions]
        if not self.img_ls:
            raise ValueError('No valid image files found (check config file).')
        if not self.output_dir.exists():
            self.logger.info(f'Creating output directory:\n{self.output_dir}')
            self.output_dir.mkdir(parents=True)

    def _load_weights(self):
        """ Invokes the model's load weights function if any weights are provided. """
        if not self.weights_path:
            raise ValueError('Weights path not specified (check config file).')
        self.logger.info(f'Loaded weights from \n > {self.weights_path}')
        self.model.model.load_weights(str(self.weights_path))
        session_config_path = self.weights_path.parent / 'session_config.yml'
        if session_config_path.exists():
            conf = yaml.load(session_config_path.read_text(), Loader=yaml.FullLoader)
        else:
            self.logger.warning('Could not find weights training configuration')
            conf = {}
        conf.update({'pre-trained-weights': self.weights_path.name})
        return conf

    def _make_save_path(self, out_folder, img_path):
        output_path = out_folder / img_path.name
        self.logger.info(f'Result in: {output_path}')
        return output_path

    def _process_image(self, img_path, out_folder):
        self.logger.info(f'Processing file\n > {img_path}')
        start = time()
        lr_img = imageio.imread(img_path)
        if lr_img.shape[2] != 3:
            raise ValueError(f'{img_path} is not an image with 3 channels.')
        sr_img = self.model.predict(lr_img)
        output_path = self._make_save_path(out_folder, img_path)
        end = time()
        self.logger.info(f'Elapsed time: {end - start:.2f}s')
        return output_path, sr_img

    def _save_image(self, output_path, sr_img):
        if output_path.exists():
            self.logger.warning('File already exists, overwriting:\n{}'.format(output_path))
        with output_path.open('wb') as f:
            imageio.imwrite(f, sr_img)

    def get_predictions(self, model, weights_path):
        """ Runs the prediction. """
        self.model = model
        self.weights_path = Path(weights_path)
        weights_conf = self._load_weights()
        out_folder = self.output_dir / self.model.name / get_timestamp()
        if out_folder.exists():
            self.logger.warning('Directory exists, might overwrite files')
        else:
            out_folder.mkdir(parents=True)
        if weights_conf:
            yaml.dump(weights_conf, (out_folder / 'weights_config.yml').open('w'))
        for img_path in self.img_ls:
            output_path, sr_img = self._process_image(img_path, out_folder)
            self._save_image(output_path, sr_img)