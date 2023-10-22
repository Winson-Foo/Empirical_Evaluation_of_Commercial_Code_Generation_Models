import logging
import copy
import os
import termcolor


class ColoredFormatter(logging.Formatter):
    """Format log levels with color"""

    COLOR_MAP = {
        'DEBUG': dict(color='green', on_color=None),
        'INFO': dict(color='cyan', on_color=None),
        'WARNING': dict(color='yellow', on_color=None),
        'ERROR': dict(color='grey', on_color='on_red'),
        'CRITICAL': dict(color='grey', on_color='on_blue'),
    }

    PREFIX = '\033['
    SUFFIX = '\033[0m'

    def format(self, record):
        """Add log ansi colors"""
        color_settings = self.COLOR_MAP.get(
            record.levelname, self.COLOR_MAP['INFO'])
        colored_message = termcolor.colored(
            record.msg, **color_settings)
        formatted_message = super().format(record)
        return colored_message + formatted_message[len(record.msg):]

def set_logger(context_name, is_verbose=False):
    """Return colored logger with specified context name and debug=is_verbose"""
    logging_handlers = logging.root.handlers
    for handler in logging_handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger(context_name)
    logger.propagate = False

    if not logger.handlers:
        logger.setLevel(logging.DEBUG if is_verbose else logging.INFO)
        formatter = ColoredFormatter(
            '%(levelname)-.1s:%(name)s:[%(filename).3s:%(funcName).3s:%(lineno)3d]:%(message)s',
            datefmt='%m-%d %H:%M:%S')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if is_verbose else logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger