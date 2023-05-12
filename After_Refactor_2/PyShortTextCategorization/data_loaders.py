import csv
import json
import os
import random
import zipfile
from collections import defaultdict
from typing import Dict

import pandas as pd


def load_csv_data(filepath: str) -> Dict[str, list]:
    """Load the training data from a CSV file.

    Load the training data from a CSV file, with the first column being the class
    labels, and second column the text data. Returns a dictionary with the class labels
    as keys, and a list of short texts as the value for each key.

    :param filepath: path of the training data (CSV)
    :return: a dictionary with class labels as keys, and lists of short texts
    """
    with open(filepath, 'r') as datafile:
        reader = csv.reader(datafile)
        headerread = False
        shorttextdict = defaultdict(list)
        for label, content in reader:
            if headerread and isinstance(content, str):
                shorttextdict[label].append(content)
            else:
                _, _ = label, content
                headerread = True
    return dict(shorttextdict)


def load_json_data(filepath: str) -> Dict[str, list]:
    """Load the training data from a JSON file.

    Load the training data from a JSON file, with the class labels as keys, and a list
    of short texts as the value for each key. Returns the corresponding dictionary.

    :param filepath: path of the training data (JSON)
    :return: a dictionary with class labels as keys, and lists of short texts
    """
    with open(filepath, 'r') as datafile:
        return json.load(datafile)