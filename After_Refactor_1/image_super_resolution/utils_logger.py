import logging
import os


def configure_logger(name: str, job_dir: str = '.') -> logging.Logger:
    """
    Configures and returns a logger that prints on stdout at INFO level and on file at DEBUG level.
    :param name: name of the logger
    :param job_dir: directory where log file will be saved
    :return: configured logger
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # configure handlers
    configure_stream_handler(logger)
    configure_file_handler(logger, job_dir)

    return logger


def configure_stream_handler(logger: logging.Logger) -> None:
    """
    Configures stream handler for logger.
    :param logger: logger to configure
    """

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)


def configure_file_handler(logger: logging.Logger, job_dir: str) -> None:
    """
    Configures file handler for logger.
    :param logger: logger to configure
    :param job_dir: directory where log file will be saved
    """

    if not logger.handlers:
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)

        fh = logging.FileHandler(filename=os.path.join(job_dir, 'log_file'))
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)