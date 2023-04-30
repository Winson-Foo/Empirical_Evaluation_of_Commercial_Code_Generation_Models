import logging
import os
import termcolor

LOG_FORMAT = '%(levelname)-.1s:%(name)s:[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s'
LOG_DATE_FORMAT = '%m-%d %H:%M:%S'

def set_logger(context, verbose=False):
    logger = logging.getLogger(context)
    logger.propagate = False
    formatter = ColoredFormatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger

class ColoredFormatter(logging.Formatter):
    MAPPING = {
        'DEBUG': dict(color='green', on_color=None),
        'INFO': dict(color='cyan', on_color=None),
        'WARNING': dict(color='yellow', on_color=None),
        'ERROR': dict(color='grey', on_color='on_red'),
        'CRITICAL': dict(color='grey', on_color='on_blue')
    }
    
    PREFIX = '\033['
    SUFFIX = '\033[0m'

    def format(self, record):
        seq = self.MAPPING.get(record.levelname, self.MAPPING['INFO'])
        msg = termcolor.colored(record.msg, **seq)
        record.msg = msg
        return super().format(record)
