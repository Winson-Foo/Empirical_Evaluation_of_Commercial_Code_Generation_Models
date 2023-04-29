from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import re

def read_label_map(path_to_labels: str) -> dict:
    '''
    Reads a label map file and returns a dictionary of id to label mappings.

    - Args:
        - path_to_labels (str): path to pbtx file

    - Returns:
        - dict of form { id(int) : label(str)}
    '''
    with open(path_to_labels, "r") as f:
        text = f.read()
    entry_pairs = []
    entry_start = text.find('item')
    while entry_start != -1:
        entry_end = text.find('item', entry_start+1)
        if entry_end == -1:
            entry_end = len(text)
        entry_text = text[entry_start:entry_end]
        entry_pairs.append(parse_label_entry(entry_text))
        entry_start = entry_end
    return {entry[0]:entry[1] for entry in entry_pairs}

def parse_label_entry(entry_text: str) -> tuple:
    '''
    Parses a single label entry from a label map file and returns a tuple of id and label.

    - Args:
        - entry_text (str): text containing a single label entry

    - Returns:
        - tuple of form (id(int), label(str))
    '''
    id_match = re.search(r'id:\s*(\d+)', entry_text)
    label_match = re.search(r'display_name:\s*[\'"]([^"\']*)[\'"]', entry_text)
    if id_match and label_match:
        return (int(id_match.group(1)), label_match.group(1))
    else:
        raise ValueError('Invalid label entry: ' + entry_text)