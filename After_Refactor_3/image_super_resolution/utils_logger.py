import logging
import os

LOG_LEVEL_INFO = logging.INFO
LOG_LEVEL_DEBUG = logging.DEBUG

LOG_FORMAT_INFO = logging.Formatter('%(message)s')
LOG_FORMAT_DEBUG = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def add_stream_handler(logger: logging.Logger, level: int = LOG_LEVEL_INFO, formatter: logging.Formatter = LOG_FORMAT_INFO) -> None:
    """ Adds a stream handler to the logger. """
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def add_file_handler(logger: logging.Logger, job_dir: str, level: int = LOG_LEVEL_DEBUG, formatter: logging.Formatter = LOG_FORMAT_DEBUG) -> None:
    """ Adds a file handler to the logger. """
    if not os.path.exists(job_dir):
        os.makedirs(job_dir)

    fh = logging.FileHandler(filename=os.path.join(job_dir, 'log_file'))
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def get_logger(name: str, job_dir: str = '.') -> logging.Logger:
    """
    Returns logger that prints on stdout at INFO level and on file at DEBUG level.
    :param name: Name of the logger.
    :param job_dir: Directory where the log file will be saved.
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL_DEBUG)
    if not logger.handlers:
        add_stream_handler(logger)
        add_file_handler(logger, job_dir)
    
    return logger