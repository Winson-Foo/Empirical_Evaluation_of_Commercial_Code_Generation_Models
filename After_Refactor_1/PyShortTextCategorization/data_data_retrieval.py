import csv
import json
import os
import random
import zipfile
from collections import defaultdict
from urllib.request import urlretrieve

import pandas as pd


def retrieve_csvdata_as_dict(filepath):
    """ Retrieve the training data in a CSV file.

    Retrieve the training data in a CSV file, with the first column being the
    class labels, and second column the text data. It returns a dictionary with
    the class labels as keys, and a list of short texts as the value for each key.

    :param filepath: path of the training data (CSV)
    :return: a dictionary with class labels as keys, and lists of short texts
    :rtype: dict
    """
    with open(filepath, 'r') as datafile:
        reader = csv.reader(datafile)
        headerread = False
        shorttextdict = defaultdict(list)
        for label, content in reader:
            if not headerread:
                category_col, descp_col = label, content
                headerread = True
                continue
            if isinstance(content, str):
                shorttextdict[label].append(content)
        return dict(shorttextdict)


def retrieve_jsondata_as_dict(filepath):
    """ Retrieve the training data in a JSON file.

    Retrieve the training data in a JSON file, with
    the class labels as keys, and a list of short texts as the value for each key.
    It returns the corresponding dictionary.

    :param filepath: path of the training data (JSON)
    :return: a dictionary with class labels as keys, and lists of short texts
    :rtype: dict
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def subjectkeywords():
    """ Return an example data set of subjects.

    Return an example data set, with three subjects and corresponding keywords.
    This is in the format of the training input.

    :return: example data set
    :rtype: dict
    """
    this_dir, _ = os.path.split(__file__)
    return retrieve_csvdata_as_dict(os.path.join(this_dir, 'shorttext_exampledata.csv'))


def inaugural():
    """ Return an example dataset, which is the Inaugural Addresses of all Presidents of 
    the United States from George Washington to Barack Obama.
    
    Each key is the year, a dash, and the last name of the president. The content is
    the list of all the sentences
    
    :return: example data set
    :rtype: dict
    """
    with zipfile.ZipFile(get_or_download_data("USInaugural.zip",
                                               "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/USInaugural.zip",
                                               asbytes=True)) as zfile:
        address_jsonstr = zfile.open("addresses.json").read()
    return json.loads(address_jsonstr.decode('utf-8'))


def nihreports(txt_col='PROJECT_TITLE', label_col='FUNDING_ICs', sample_size=512):
    """ Return an example data set, sampled from NIH RePORT (Research Portfolio
    Online Reporting Tools).

    Return an example data set from NIH (National Institutes of Health),
    data publicly available from their RePORT
    website. (`link
    <https://exporter.nih.gov/ExPORTER_Catalog.aspx>`_).
    The data is with `txt_col` being either project titles ('PROJECT_TITLE')
    or proposal abstracts ('ABSTRACT_TEXT'), and label_col being the names of the ICs (Institutes or Centers),
    with 'IC_NAME' the whole form, and 'FUNDING_ICs' the abbreviated form).

    Dataset directly adapted from the NIH data from `R` package `textmineR
    <https://cran.r-project.org/web/packages/textmineR/index.html>`_.

    :param txt_col: column for the text (Default: 'PROJECT_TITLE')
    :param label_col: column for the labels (Default: 'FUNDING_ICs')
    :param sample_size: size of the sample. Set to None if all rows. (Default: 512)
    :return: example data set
    :type txt_col: str
    :type label_col: str
    :type sample_size: int
    :rtype: dict
    """
    # validation
    # txt_col = 'PROJECT_TITLE' or 'ABSTRACT_TEXT'
    # label_col = 'FUNDING_ICs' or 'IC_NAME'
    if txt_col not in ['PROJECT_TITLE', 'ABSTRACT_TEXT']:
        raise KeyError(f"Undefined text column: {txt_col}. Must be PROJECT_TITLE or ABSTRACT_TEXT.")
    if label_col not in ['FUNDING_ICs', 'IC_NAME']:
        raise KeyError(f"Undefined label column: {label_col}. Must be FUNDING_ICs or IC_NAME.")

    with zipfile.ZipFile(get_or_download_data('nih_full.csv.zip',
                                               'https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/nih_full.csv.zip',
                                               asbytes=True),
                         'r',
                         zipfile.ZIP_DEFLATED) as zfile:
        nih = pd.read_csv(zfile.open('nih_full.csv'),
                           na_filter=False,
                           usecols=[label_col, txt_col],
                           encoding='cp437')

    nb_data = len(nih)
    sample_size = nb_data if sample_size is None else min(nb_data, sample_size)

    classdict = defaultdict(list)

    for rowidx in random.sample(range(nb_data), sample_size):
        label = nih.iloc[rowidx, nih.columns.get_loc(label_col)]
        if label_col == 'FUNDING_ICs':
            if not label:
                label = 'OTHER'
            else:
                endpos = label.index(':')
                label = label[:endpos]
        content = nih.iloc[rowidx, nih.columns.get_loc(txt_col)]
        if isinstance(content, str):
            classdict[label].append(content)

    return dict(classdict)


def mergedict(dicts):
    """ Merge data dictionary.

    Merge dictionaries of the data in the training data format.

    :param dicts: dicts to merge
    :return: merged dict
    :type dicts: list
    :rtype: dict
    """
    mdict = defaultdict(list)
    for thisdict in dicts:
        for label in thisdict:
            mdict[label] += thisdict[label]
    return dict(mdict)


def yield_crossvalidation_classdicts(classdict, nb_partitions, shuffle=False):
    """ Yielding test data and training data for cross validation by partitioning it.

    Given a training data, partition the data into portions, each will be used as test
    data set, while the other training data set. It returns a generator.

    :param classdict: training data
    :param nb_partitions: number of partitions
    :param shuffle: whether to shuffle the data before partitioning
    :return: generator, producing a test data set and a training data set each time
    :type classdict: dict
    :type nb_partitions: int
    :type shuffle: bool
    :rtype: generator
    """
    crossvaldicts = [defaultdict(list) for _ in range(nb_partitions)]

    for label in classdict:
        nb_data = len(classdict[label])
        partsize = nb_data // nb_partitions
        sentences = classdict[label]
        if shuffle:
            random.shuffle(sentences)
        for i in range(nb_partitions):
            crossvaldicts[i][label].extend(sentences[i * partsize: min(nb_data, (i + 1) * partsize)])
    crossvaldicts = [dict(crossvaldict) for crossvaldict in crossvaldicts]

    for i, testdict in enumerate(crossvaldicts):
        traindict = mergedict([crossvaldicts[j] for j in range(nb_partitions) if j != i])
        yield testdict, traindict


def get_or_download_data(filename, origin, asbytes=False):
    # determine path
    homedir = os.path.expanduser('~')
    datadir = os.path.join(homedir, '.shorttext')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    targetfilepath = os.path.join(datadir, filename)
    # download if not exist
    if not os.path.exists(targetfilepath):
        print('Downloading...')
        print('Source: ', origin)
        print('Target: ', targetfilepath)
        try:
            urlretrieve(origin, targetfilepath)
        except Exception as e:
            print(f"Failure to download file {filename}: {e}")
            os.remove(targetfilepath)

    # return
    return open(targetfilepath, 'rb' if asbytes else 'r')