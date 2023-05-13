# File: parsing.py

import re

def escape_string(s):
    return re.sub('\W', '_', s)