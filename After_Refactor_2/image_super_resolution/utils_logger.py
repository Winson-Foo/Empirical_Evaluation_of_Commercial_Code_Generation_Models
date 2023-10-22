import logging
import os


def configure_file_handler(logger, job_dir='.'):
    """ Configures a file handler for the logger. """
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    fh = logging.FileHandler(filename=os.path.join(job_dir, 'log_file'))
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)


def configure_console_handler(logger):
    """ Configures a console (stdout) handler for the logger. """
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)


def get_logger(name, job_dir='.'):
    """ Returns logger that prints on stdout at INFO level and on file at DEBUG level. """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        configure_console_handler(logger)
        configure_file_handler(logger, job_dir)

    return logger