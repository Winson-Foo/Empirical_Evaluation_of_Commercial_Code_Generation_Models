from pathlib import Path
from time import time
from typing import List, Tuple, Union

import imageio
import numpy as np
import yaml

from ISR.models import RDN
from ISR.utils.logger import get_logger
from ISR.utils.utils import get_timestamp

ACCEPTED_EXTENSIONS = ('.jpeg', '.jpg', '.png')
WEIGHTS_CONFIG_FILE_NAME = 'weights_config.yml'
SESSION_CONFIG_FILE_NAME = 'session_config.yml'

class Predictor:
    """ The predictor class handles prediction, given an input model.

    Loads the images in the input directory, executes training given a model
    and saves the results in the output directory.
    Can receive a path for the weights or can let the user browse through the
    weights directory for the desired weights.

    Attributes:
        input_dir (str): The path to the input directory.
        output_dir (str): The path to the output directory.
        verbose (bool): A flag indicating whether to print logger messages.

    Examples:
        >>> predictor = Predictor(input_dir='input', output_dir='output', verbose=True)
        >>> model = RDN(weights='psnr-small')
        >>> predictor.get_predictions(model)

    """

    def __init__(self, input_dir: str, output_dir: str = './data/output', verbose: bool = True) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) / self.input_dir.name
        self.logger = get_logger(__name__)
        if not verbose:
            self.logger.setLevel(40)
        self.extensions = ACCEPTED_EXTENSIONS
        self.image_list = self._get_image_list()
        self.create_output_dir()

    def _get_image_list(self) -> List[Path]:
        """ Retrieves a list of image files in the input directory. """
        image_list = [f for f in self.input_dir.iterdir() if f.suffix in self.extensions]
        if len(image_list) < 1:
            self.logger.error(f'No valid image files found in {self.input_dir}')
            raise ValueError(f'No valid image files found in {self.input_dir}')
        return image_list
    
    def create_output_dir(self) -> None:
        """ Creates the output directory if it does not exist. """
        if not self.output_dir.exists():
            self.logger.info(f'Creating output directory: {self.output_dir}')
            self.output_dir.mkdir(parents=True)

    def _load_weights(self, weights_path: Path) -> dict:
        """ Loads the model weights given the path to the weights. """
        if not weights_path.exists():
            self.logger.error(f'Weights not found at {weights_path}')
            raise ValueError(f'Weights not found at {weights_path}')
        
        self.logger.info(f'Loaded weights from {weights_path}')
        self.model.model.load_weights(str(weights_path))

        session_config_path = weights_path.parent / SESSION_CONFIG_FILE_NAME
        if session_config_path.exists():
            conf = yaml.load(session_config_path.read_text(), Loader=yaml.FullLoader)
        else:
            self.logger.warning('Could not find weights training configuration')
            conf = {}
        
        conf.update({'pre-trained-weights': weights_path.name})
        return conf

    def _make_basename(self, model: RDN) -> str:
        """ Combines the model's architecture name with its parameters. """
        params = [model.name]
        for param in np.sort(list(model.params.keys())):
            params.append(f'{param}{model.params[param]}')
        return '-'.join(params)

    def get_predictions(self, model: RDN, weights_path: Union[str, Path]) -> None:
        """ Runs the predictions using the specified model and saves the results to the output directory. """
        weights_path = Path(weights_path)
        weights_conf = self._load_weights(weights_path)
        out_folder = self.output_dir / self._make_basename(model) / get_timestamp()

        self.logger.info(f'Results in: {out_folder}')

        if out_folder.exists():
            self.logger.warning('Directory exists, might overwrite files')
        else:
            out_folder.mkdir(parents=True)

        if weights_conf:
            yaml.dump(weights_conf, (out_folder / WEIGHTS_CONFIG_FILE_NAME).open('w'))

        for img_path in self.image_list:
            output_path = out_folder / img_path.name
            self.logger.info(f'Processing file: {img_path}')
            start = time()
            sr_img = self._forward_pass(model, img_path)
            end = time()
            self.logger.info(f'Elapsed time: {end - start}s')
            self.logger.info(f'Result in: {output_path}')
            imageio.imwrite(output_path, sr_img)

    def _forward_pass(self, model: RDN, file_path: Path) -> np.ndarray:
        """ Processes the image file using the given model. """
        lr_img = imageio.imread(file_path)
        if not lr_img.shape[2] == 3:
            self.logger.error(f'{file_path} is not an image with 3 channels.')
            raise ValueError(f'{file_path} is not an image with 3 channels.')
        return model.predict(lr_img)