# data_utils.py

import os
import json
import csv
import zipfile
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd


def retrieve_csv_data(filepath: str) -> Dict[str, List[str]]:
    """Parse a CSV file with one column for labels and
    one column for text content.
    The function returns a dictionary with the labels as keys and a
    list of text content as the value for each key."""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header_read = False
        output_dict = defaultdict(lambda: [])
        for label, content in reader:
            if header_read:
                if isinstance(content, str):
                    output_dict[label] += [content]
            else:
                category_col, descp_col = label, content
                header_read = True
    return dict(output_dict)


def retrieve_json_data(filepath: str) -> Dict[str, List[str]]:
    """Load a JSON file with labels as keys and lists of text content as values."""
    with open(filepath, 'r') as f:
        return json.load(f)


def merge_dicts(dicts: List[Dict[str, List[str]]]) -> Dict[str, List[str]]:
    """Merge a list of dictionaries containing training data."""
    merged_dict = defaultdict(lambda: [])
    for this_dict in dicts:
        for label in this_dict:
            merged_dict[label] += this_dict[label]
    return dict(merged_dict)


def yield_crossvalidated_dicts(class_dict: Dict[str, List[str]], nb_partitions: int,
                               shuffle: bool = False) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Partition training data into test sets and training sets for cross validation.
    The input `class_dict` contains one key for each class label and a list of
    examples as the value for each key.
    The output from the function is a tuple with test_dict as the first element,
    which contains a subset of `class_dict`, and train_dict as the second element,
    which contains the rest of the examples in `class_dict`."""
    cross_val_dicts = []
    for _ in range(nb_partitions):
        cross_val_dicts.append(defaultdict(lambda: []))

    for label in class_dict:
        nb_data = len(class_dict[label])
        part_size = nb_data / nb_partitions
        sentences = class_dict[label] if not shuffle else sorted(class_dict[label])
        for i in range(nb_partitions):
            cross_val_dicts[i][label] += sentences[i * part_size:min(nb_data, (i + 1) * part_size)]
    cross_val_dicts = [dict(cross_val_dict) for cross_val_dict in cross_val_dicts]

    for i in range(nb_partitions):
        test_dict = cross_val_dicts[i]
        train_dict = merge_dicts([cross_val_dicts[j] for j in range(nb_partitions) if j != i])
        yield test_dict, train_dict


def download_data(filename: str, origin: str, target_dir: str) -> str:
    """Download a file from a URL to a target directory if it does not exist."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    target_filepath = os.path.join(target_dir, filename)
    if not os.path.exists(target_filepath):
        print('Downloading...')
        print('Source: ', origin)
        print('Target: ', target_filepath)
        try:
            urlretrieve(origin, target_filepath)
        except:
            print('Failure to download file!')
            print(sys.exc_info())
            os.remove(target_filepath)

    return target_filepath


def retrieve_inaugural_data() -> Dict[str, List[str]]:
    """Retrieve the dataset containing the Inaugural Addresses of all
    Presidents of the United States from George Washington to Barack Obama."""
    target_dir = os.path.join(os.path.expanduser('~'), '.shorttext')
    filepath = download_data("USInaugural.zip",
                             "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/USInaugural.zip",
                             target_dir)

    with zipfile.ZipFile(filepath) as zf:
        address_jsonstr = zf.read("addresses.json").decode('utf-8')

    return json.loads(address_jsonstr)


def retrieve_subject_keywords_data() -> Dict[str, List[str]]:
    """Return an example data set of subjects with corresponding keywords."""
    this_dir = os.path.dirname(__file__)
    filepath = os.path.join(this_dir, 'shorttext_exampledata.csv')
    return retrieve_csv_data(filepath)


def retrieve_nih_reports_data(txt_col: str = 'PROJECT_TITLE', label_col: str = 'FUNDING_ICs',
                              sample_size: int = 512) -> Dict[str, List[str]]:
    """Return a sample data set from NIH (National Institutes of Health) RePORT."""
    # validate input
    if txt_col not in ['PROJECT_TITLE', 'ABSTRACT_TEXT']:
        raise KeyError('Invalid text column: ' + txt_col + '. Must be PROJECT_TITLE or ABSTRACT_TEXT.')
    if label_col not in ['FUNDING_ICs', 'IC_NAME']:
        raise KeyError('Invalid label column: ' + label_col + '. Must be FUNDING_ICs or IC_NAME.')

    target_dir = os.path.join(os.path.expanduser('~'), '.shorttext')
    filepath = download_data('nih_full.csv.zip',
                             'https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/nih_full.csv.zip',
                             target_dir)

    nih = pd.read_csv(filepath, na_filter=False, usecols=[label_col, txt_col], encoding='cp437')
    nb_data = len(nih)
    sample_size = nb_data if sample_size is None else min(nb_data, sample_size)
    data_dict = defaultdict(lambda: [])

    for rowidx in range(sample_size):
        label = nih.iloc[rowidx, nih.columns.get_loc(label_col)]
        if label_col == 'FUNDING_ICs':
            label = label.split(':')[0] if label else 'OTHER'
        data_dict[label] += [nih.iloc[rowidx, nih.columns.get_loc(txt_col)]]

    return dict(data_dict)