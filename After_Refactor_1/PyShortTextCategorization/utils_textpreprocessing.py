import re
import os
import codecs

import snowballstemmer


def tokenize_text(text: str) -> list:
    """Tokenize the text by splitting it at whitespaces."""
    return text.split()


def stem_word(word: str) -> str:
    """Stem the given word using Snowball Stemmer's Porter stemmer."""
    stemmer = snowballstemmer.stemmer('porter')
    return stemmer.stemWord(word)


def preprocess_text(text: str, pipeline: list) -> str:
    """Preprocess the text according to the given pipeline.

    Given the pipeline, which is a list of functions that process an
    input text to another text (e.g., stemming, lemmatizing, removing punctuations etc.),
    preprocess the text.

    Args:
        text: Text string to be preprocessed
        pipeline: List of functions that process the input and convert it to other text

    Returns:
        Preprocessed text

    """
    if not pipeline:
        return text
    return preprocess_text(pipeline[0](text), pipeline[1:])


def text_preprocessor(pipeline: list) -> callable:
    """Return the function that preprocesses text according to the pipeline.

    Given the pipeline, which is a list of functions that process an
    input text to another text (e.g., stemming, lemmatizing, removing punctuations etc.),
    return a function that preprocesses an input text outlined by the pipeline, essentially
    a function that runs :func:`~preprocess_text` with the specified pipeline.

    Args:
        pipeline: List of functions that process the input and convert it to other text

    Returns:
        Function that preprocesses the input text according to the given pipeline

    """
    def preprocess_input_text(text: str) -> str:
        """Preprocess the input text according to the given pipeline."""
        return preprocess_text(text, pipeline)

    return preprocess_input_text


def load_stopwords(file_path: str) -> set:
    """Load the stop words list from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = {word.strip() for word in f}

    return stopwords


def get_oldschool_standard_text_preprocessor(stopwords_file_path: str) -> callable:
    """Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words, and
    - stemming the words (using Porter stemmer).

    This function calls :func:`~text_preprocessor`.

    Args:
        stopwords_file_path: File path of the list of stop words

    Returns:
        Function that preprocesses the input text according to the pipeline
    """
    stopwords = load_stopwords(stopwords_file_path)

    # the pipeline
    pipeline = [
        lambda text: re.sub('[^\w\s]', '', text),
        lambda text: re.sub('[\d]', '', text),
        lambda text: text.lower(),
        lambda text: [word for word in tokenize_text(text) if word not in stopwords],
        lambda text: ' '.join([stem_word(word) for word in tokenize_text(text)])
    ]

    return text_preprocessor(pipeline)


def get_standard_text_preprocessor_1() -> callable:
    """Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words (NLTK list), and
    - stemming the words (using Porter stemmer).

    This function calls :func:`~get_oldschool_standard_text_preprocessor`.

    Returns:
        Function that preprocesses the input text according to the pipeline
    """
    # load stop words
    this_dir, _ = os.path.split(__file__)
    stopwords_file_path = os.path.join(this_dir, 'stopwords.txt')

    return get_oldschool_standard_text_preprocessor(stopwords_file_path)


def get_standard_text_preprocessor_2() -> callable:
    """Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words (NLTK list minus negation terms), and
    - stemming the words (using Porter stemmer).

    This function calls :func:`~get_oldschool_standard_text_preprocessor`.

    Returns:
        Function that preprocesses the input text according to the pipeline
    """
    # load stop words
    this_dir, _ = os.path.split(__file__)
    stopwords_file_path = os.path.join(this_dir, 'nonneg_stopwords.txt')

    return get_oldschool_standard_text_preprocessor(stopwords_file_path)