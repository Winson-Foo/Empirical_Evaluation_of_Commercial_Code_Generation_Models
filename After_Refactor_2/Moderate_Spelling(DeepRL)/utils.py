# utils.py
import os
import logging


def get_logger(tag='', log_level=logging.DEBUG):
    logger = logging.getLogger(tag)
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return logger


def close_obj(obj):
    if hasattr(obj, 'close'):
        obj.close()


def mkdir(path):
    os.makedirs(path, exist_ok=True)