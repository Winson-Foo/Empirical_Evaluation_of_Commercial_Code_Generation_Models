import os
import zipfile
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import pandas as pd
from .data_loaders import load_csv_data


def load_subject_keywords() -> Dict[str, list]:
    """Return an example data set of subjects.

    Return an example data set, with three subjects and corresponding keywords. This
    is in the format of the training input.

    :return: example data set
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = 'shorttext_exampledata.csv'
    filepath = os.path.join(dir_path, filename)
    return load_csv_data(filepath)


def load_inaugural_addresses() -> Dict[str, list]:
    """Return an example dataset, which is the Inaugural Addresses of all
    Presidents of the United States from George Washington to Barack Obama.

    Each key is the year, a dash, and the last name of the president. The content is
    the list of all the sentences.

    :return: example data set
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = 'USInaugural.zip'
    filepath = os.path.join(dir_path, filename)
    zfile = zipfile.ZipFile(filepath)
    address_jsonstr = zfile.open("addresses.json").read()
    zfile.close()
    return json.loads(address_jsonstr)


def load_nih_reports(txt_col: str = 'PROJECT_TITLE', label_col: str = 'FUNDING_ICs',
                      sample_size: Optional[int] = 512) -> Dict[str, list]:
    """ Return an example data set, sampled from NIH RePORT (Research Portfolio Online
    Reporting Tools).

    Return an example data set from NIH (National Institutes of Health), data publicly
    available from their RePORT website. (`link
    <https://exporter.nih.gov/ExPORTER_Catalog.aspx>`_).
    The data is with `txt_col` being either project titles ('PROJECT_TITLE') or proposal
    abstracts ('ABSTRACT_TEXT'), and label_col being the names of the ICs (Institutes or
    Centers), with 'IC_NAME' the whole form, and 'FUNDING_ICs' the abbreviated form).

    Dataset directly adapted from the NIH data from `R` package `textmineR
    <https://cran.r-project.org/web/packages/textmineR/index.html>`_.

    :param txt_col: column for the text (Default: 'PROJECT_TITLE')
    :param label_col: column for the labels (Default: 'FUNDING_ICs')
    :param sample_size: size of the sample. Set to None if all rows. (Default: 512)
    :return: example data set
    """
    if txt_col not in ['PROJECT_TITLE', 'ABSTRACT_TEXT']:
        raise KeyError(f"Undefined text column: {txt_col}. Must be PROJECT_TITLE or ABSTRACT_TEXT.")
    if label_col not in ['FUNDING_ICs', 'IC_NAME']:
        raise KeyError(f"Undefined label column: {label_col}. Must be FUNDING_ICs or IC_NAME.")

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = 'nih_full.csv.zip'
    filepath = os.path.join(dir_path, filename)
    zfile = zipfile.ZipFile(filepath, 'r', zipfile.ZIP_DEFLATED)
    nih = pd.read_csv(zfile.open('nih_full.csv'), na_filter=False, usecols=[label_col, txt_col], encoding='cp437')
    zfile.close()
    nb_data = len(nih)
    sample_size = nb_data if sample_size is None else min(nb_data, sample_size)

    classdict = defaultdict(list)

    for rowidx in np.random.randint(nb_data, size=min(nb_data, sample_size)):
        label = nih.iloc[rowidx, nih.columns.get_loc(label_col)]
        if label_col == 'FUNDING_ICs':
            if label == '':
                label = 'OTHER'
            else:
                endpos = label.index(':')
                label = label[:endpos]
        classdict[label].append(nih.iloc[rowidx, nih.columns.get_loc(txt_col)])

    return dict(classdict)


def merge_data_dicts(dicts: list[Dict[str, list]]) -> Dict[str, list]:
    """Merge dictionaries containing the training data.

    Merge dictionaries containing the training data in the format of class labels as
    keys and lists of short texts as values.

    :param dicts: dictionaries to merge
    :return: merged dictionary
    """
    mdict = defaultdict(list)
    for thisdict in dicts:
        for label in thisdict:
            mdict[label].extend(thisdict[label])
    return dict(mdict)


def yield_cross_validation_data_dicts(classdict: Dict[str, list], nb_partitions: int,
                                      shuffle: bool = False) -> tuple[Dict[str, list], Dict[str, list]]:
    """Yield test data and training data for cross validation by partitioning it.

    Given the training data, partition the data into portions, each of which will be
    used as a test data set, and the others as training data sets. It returns a
    generator of tuples, each containing a test data set and a corresponding training
    data set.

    :param classdict: training data
    :param nb_partitions: number of partitions
    :param shuffle: whether to shuffle the data before partitioning
    :return: generator, producing a tuple of a test data set and a training data set each time
    """
    crossvaldicts = [defaultdict(list) for _ in range(nb_partitions)]
    for label in classdict:
        nb_data = len(classdict[label])
        part_size = nb_data // nb_partitions
        sentences = classdict[label]
        if shuffle:
            random.shuffle(sentences)
        for i in range(nb_partitions):
            start_pos = i * part_size
            end_pos = min(nb_data, (i + 1) * part_size)
            crossvaldicts[i][label].extend(sentences[start_pos:end_pos])

    crossvaldicts = [dict(crossvaldict) for crossvaldict in crossvaldicts]

    for i in range(nb_partitions):
        testdict = crossvaldicts[i]
        traindicts = [crossvaldicts[j] for j in range(nb_partitions) if j != i]
        traindict = merge_data_dicts(traindicts)
        yield testdict, traindict