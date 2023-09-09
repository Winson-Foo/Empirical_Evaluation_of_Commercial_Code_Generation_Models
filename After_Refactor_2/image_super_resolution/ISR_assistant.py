import os
from importlib import import_module
from typing import Dict, Any

import numpy as np

from ISR.utils import utils
from ISR.utils.logger import get_logger

def load_generator_module(generator: str) -> Any:
    """Load the specified generator module."""
    return import_module(f"ISR.models.{generator}")

def run_program(config_file: str, default: bool = False, training: bool = False, 
                prediction: bool = False) -> None:
    """Run the ISR program."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger = get_logger(__name__)

    # Load the configuration and data for the specified session type
    session_type, generator, config, dataset = utils.setup(config_file, default, training, prediction)
    lr_patch_size = config['session'][session_type]['patch_size']
    scale = config['generators'][generator]['x']

    # Load the generator model
    generator_module = load_generator_module(generator)
    gen = generator_module.make_model(config['generators'][generator], lr_patch_size)

    # Run the prediction or training session
    if session_type == 'prediction':
        from ISR.predict.predictor import Predictor
        predictor = Predictor(input_dir=config['test_sets'][dataset])
        predictor.get_predictions(gen, config['weights_paths']['generator'])

    elif session_type == 'training':
        # Load the feature extractor and discriminator models, if specified
        hr_patch_size = lr_patch_size * scale
        f_ext = None
        if config['default']['feature_extractor']:
            from ISR.models.cut_vgg19 import Cut_VGG19
            out_layers = config['feature_extractor']['vgg19']['layers_to_extract']
            f_ext = Cut_VGG19(patch_size=hr_patch_size, layers_to_extract=out_layers)

        discr = None
        if config['default']['discriminator']:
            from ISR.models.discriminator import Discriminator
            discr = Discriminator(patch_size=hr_patch_size, kernel_size=3)

        # Train the model
        trainer = Trainer(
            generator=gen,
            discriminator=discr,
            feature_extractor=f_ext,
            lr_train_dir=config['training_sets'][dataset]['lr_train_dir'],
            hr_train_dir=config['training_sets'][dataset]['hr_train_dir'],
            lr_valid_dir=config['training_sets'][dataset]['lr_valid_dir'],
            hr_valid_dir=config['training_sets'][dataset]['hr_valid_dir'],
            learning_rate=config['session'][session_type]['learning_rate'],
            loss_weights=config['loss_weights'],
            losses=config['losses'],
            dataname=config['training_sets'][dataset]['data_name'],
            log_dirs=config['log_dirs'],
            weights_generator=config['weights_paths']['generator'],
            weights_discriminator=config['weights_paths']['discriminator'],
            n_validation=config['session'][session_type]['n_validation_samples'],
            flatness=config['session'][session_type]['flatness'],
            fallback_save_every_n_epochs=config['session'][session_type]['fallback_save_every_n_epochs'],
            adam_optimizer=config['session'][session_type]['adam_optimizer'],
            metrics=config['session'][session_type]['metrics'],
        )
        trainer.train(
            epochs=config['session'][session_type]['epochs'],
            steps_per_epoch=config['session'][session_type]['steps_per_epoch'],
            batch_size=config['session'][session_type]['batch_size'],
            monitored_metrics=config['session'][session_type]['monitored_metrics'],
        )
    else:
        logger.error('Invalid session type specified.')


if __name__ == '__main__':
    args: Dict[str, Any] = utils.parse_args()
    np.random.seed(1000)
    run_program(
        config_file=args['config_file'],
        default=args['default'],
        training=args['training'],
        prediction=args['prediction'],
    )