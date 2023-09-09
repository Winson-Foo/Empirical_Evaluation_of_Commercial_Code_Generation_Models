import os
import numpy as np
from ISR.utils.logger import get_logger
from ISR.utils.config import read_config

def parse_args():
    # TODO: implement command line argument parser
    pass

def setup_session(session_type, config, default=False, training=False, prediction=False):
    lr_patch_size = config['session'][session_type]['patch_size']
    scale = config['generators'][generator]['x']
    
    module = import_module('ISR.models.' + generator)
    gen = module.make_model(config['generators'][generator], lr_patch_size)
    
    if session_type == 'prediction':
        return setup_prediction(config, gen)
    elif session_type == 'training':
        return setup_training(config, gen)
    else:
        raise ValueError('Invalid session type %s' % session_type)

def setup_prediction(config, generator):
    from ISR.predict.predictor import Predictor
    pr_h = Predictor(input_dir=config['test_sets'][dataset])
    pr_h.get_predictions(generator, config['weights_paths']['generator'])

def setup_training(config, generator):
    hr_patch_size = lr_patch_size * scale
    f_ext = create_feature_extractor(config)
    discr = create_discriminator(config)
    
    trainer = Trainer(
        generator=gen,
        discriminator=discr,
        feature_extractor=f_ext,
        lr_train_dir=config['training_sets'][dataset]['lr_train_dir'],
        hr_train_dir=config['training_sets'][dataset]['hr_train_dir'],
        lr_valid_dir=config['training_sets'][dataset]['lr_valid_dir'],
        hr_valid_dir=config['training_sets'][dataset]['hr_valid_dir'],
        learning_rate=config['session']['training']['learning_rate'],
        loss_weights=config['loss_weights'],
        losses=config['losses'],
        dataname=config['training_sets'][dataset]['data_name'],
        log_dirs=config['log_dirs'],
        weights_generator=config['weights_paths']['generator'],
        weights_discriminator=config['weights_paths']['discriminator'],
        n_validation=config['session']['training']['n_validation_samples'],
        flatness=config['session']['training']['flatness'],
        fallback_save_every_n_epochs=config['session']['training']['fallback_save_every_n_epochs'],
        adam_optimizer=config['session']['training']['adam_optimizer'],
        metrics=config['session']['training']['metrics']
    )
    return trainer

def create_discriminator(config):
    if config['default']['discriminator']:
        from ISR.models.discriminator import Discriminator
        discr = Discriminator(patch_size=hr_patch_size, kernel_size=3)
        return discr
    else:
        return None

def create_feature_extractor(config):
    if config['default']['feature_extractor']:
        from ISR.models.cut_vgg19 import Cut_VGG19
        out_layers = config['feature_extractor']['vgg19']['layers_to_extract']
        f_ext = Cut_VGG19(patch_size=hr_patch_size, layers_to_extract=out_layers)
        return f_ext
    else:
        return None

def run(config_file, default=False, training=False, prediction=False):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logger = get_logger(__name__)
    config = read_config(config_file)
    session_type, generator, dataset = config['session'], config['generators'], config['dataset']
    session = setup_session(session_type, config, default, training, prediction)
    session.run()

    else:
        logger.error('Invalid choice.')

if __name__ == '__main__':
    np.random.seed(1000)
    args = parse_args()
    run(config_file=args['config_file'], default=args['default'], training=args['training'], prediction=args['prediction'])