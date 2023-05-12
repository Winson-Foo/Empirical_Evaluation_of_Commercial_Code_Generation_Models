import re
import os
import codecs

import snowballstemmer

from typing import List

stemmer = snowballstemmer.stemmer('porter')


def tokenize(s: str) -> List[str]:
    """Tokenize a string by splitting on whitespace.

    :param s: input string
    :return: list of tokens
    """
    return s.split()


def preprocess_text(text: str, pipeline: List[callable]) -> str:
    """Preprocess the text according to the given pipeline.

    Given the pipeline, which is a list of functions that process an
    input text to another text (e.g., stemming, lemmatizing, removing punctuations etc.),
    preprocess the text.

    :param text: text to be preprocessed
    :param pipeline: a list of functions that convert a text to another text
    :return: preprocessed text
    """
    if not pipeline:
        return text
    return preprocess_text(pipeline[0](text), pipeline[1:])


def text_preprocessor(pipeline: List[callable]) -> callable:
    """Return the function that preprocesses text according to the pipeline.

    Given the pipeline, which is a list of functions that process an
    input text to another text (e.g., stemming, lemmatizing, removing punctuations etc.),
    return a function that preprocesses an input text outlined by the pipeline, essentially
    a function that runs preprocess_text with the specified pipeline.

    :param pipeline: a list of functions that convert a text to another text
    :return: a function that preprocesses text according to the pipeline
    """
    def preprocess(text: str) -> str:
        return preprocess_text(text, pipeline)
    return preprocess


def oldschool_standard_text_preprocessor(stopwordsfile: str) -> callable:
    """Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words, and
    - stemming the words (using Porter stemmer).

    This function calls text_preprocessor.

    :param stopwordsfile: path to the file containing the list of stop words
    :return: a function that preprocesses text according to the pipeline
    """
    # load stop words file
    with codecs.open(stopwordsfile, 'r', 'utf-8') as f:
        stopwordset = set([stopword.strip() for stopword in f])

    # the pipeline
    pipeline = [
        lambda s: re.sub('[^\w\s]', '', s),
        lambda s: re.sub('[\d]', '', s),
        lambda s: s.lower(),
        lambda s: ' '.join(filter(lambda s: not (s in stopwordset), tokenize(s))),
        lambda s: ' '.join([stemmer.stemWord(stemmed_token) for stemmed_token in tokenize(s)])
    ]
    return text_preprocessor(pipeline)


def standard_text_preprocessor_1() -> callable:
    """Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words (NLTK list), and
    - stemming the words (using Porter stemmer).

    This function calls oldschool_standard_text_preprocessor.

    :return: a function that preprocesses text according to the pipeline
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    stopwordsfile = os.path.join(this_dir, 'stopwords.txt')
    return oldschool_standard_text_preprocessor(stopwordsfile)


def standard_text_preprocessor_2() -> callable:
    """Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words (NLTK list minus negation terms), and
    - stemming the words (using Porter stemmer).

    This function calls oldschool_standard_text_preprocessor.

    :return: a function that preprocesses text according to the pipeline
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    stopwordsfile = os.path.join(this_dir, 'nonneg_stopwords.txt')
    return oldschool_standard_text_preprocessor(stopwordsfile)