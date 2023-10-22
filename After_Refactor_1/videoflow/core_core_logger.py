# logger.py
import logging
from .constants import LOGGING_LEVEL

logg = logging.getLogger(__name__)
logg.setLevel(LOGGING_LEVEL)
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logg.addHandler(ch)
