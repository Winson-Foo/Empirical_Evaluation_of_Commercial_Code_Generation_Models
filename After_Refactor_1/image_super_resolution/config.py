import configparser
import os

conf = configparser.ConfigParser()

conf['default'] = {
    'feature_extractor': False,
    'discriminator': False,
    'generator': 'rdn',
    'training_set': 'test',
    'test_set': 'test',
}

conf['session'] = {}
conf['session']['training'] = {}
conf['session']['training']['patch_size'] = 0
conf['session']['training']['epochs'] = 0
conf['session']['training']['steps_per_epoch'] = 0
conf['session']['training']['batch_size'] = 0
conf['session']['prediction'] = {}
conf['session']['prediction']['patch_size'] = 5
conf['generators'] = {}
conf['generators']['rdn'] = {}
conf['generators']['rdn']['x'] = 0
conf['training_sets'] = {}
conf['training_sets']['test'] = {}
conf['training_sets']['test']['lr_train_dir'] = None
conf['training_sets']['test']['hr_train_dir'] = None
conf['training_sets']['test']['lr_valid_dir'] = None
conf['training_sets']['test']['hr_valid_dir'] = None
conf['loss_weights'] = None
conf['training_sets']['test']['data_name'] = None
conf['log_dirs'] = {}
conf['log_dirs']['logs'] = None
conf['log_dirs']['weights'] = None
conf['weights_paths'] = {}
conf['weights_paths']['generator'] = 'a/path/rdn-C1-D6-G1-G02-x0-weights.hdf5'
conf['weights_paths']['discriminator'] = 'a/path/rdn-weights.hdf5'
conf['session']['training']['n_validation_samples'] = None
conf['session']['training']['metrics'] = None
conf['session']['training']['learning_rate'] = {}
conf['session']['training']['adam_optimizer'] = None
conf['session']['training']['flatness'] = None
conf['session']['training']['fallback_save_every_n_epochs'] = None
conf['session']['training']['monitored_metrics'] = None
conf['losses'] = None

def load_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"{config_file} not found")
    conf.read(config_file)