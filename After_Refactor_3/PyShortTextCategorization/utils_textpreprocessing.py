import re
import os
import codecs
from typing import List

import snowballstemmer


def tokenize(text: str) -> List[str]:
    """
    Split the given text into tokens.

    :param text: Text to be tokenized
    :return: List of tokens
    """
    return text.split(' ')


def stem_word(word: str, stemmer: snowballstemmer.stemmer) -> str:
    """
    Stem the given word using the given stemmer.

    :param word: Word to be stemmed
    :param stemmer: Stemmer object to be used for stemming
    :return: Stemmed word
    """
    return stemmer.stemWord(word)


def remove_special_characters(text: str) -> str:
    """
    Remove special characters from the given text.

    :param text: Text to be processed
    :return: Text with special characters removed
    """
    return re.sub('[^\w\s]', '', text)


def remove_numerals(text: str) -> str:
    """
    Remove numerals from the given text.

    :param text: Text to be processed
    :return: Text with numerals removed
    """
    return re.sub('[\d]', '', text)


def convert_to_lower_case(text: str) -> str:
    """
    Convert all alphabets in the given text to lower case.

    :param text: Text to be processed
    :return: Text with all alphabets converted to lower case
    """
    return text.lower()


def remove_stopwords(text: str, stopword_set: set) -> str:
    """
    Remove stopwords from the given text using the given stopword set.

    :param text: Text to be processed
    :param stopword_set: Set of stopwords to be removed from the given text
    :return: Text with stopwords removed
    """
    return ' '.join(filter(lambda s: not (s in stopword_set), tokenize(text)))


def preprocess_text(text: str, pipeline: List) -> str:
    """
    Preprocess the given text using the given pipeline.

    :param text: Text to be preprocessed
    :param pipeline: List of functions to be applied on the given text for preprocessing
    :return: Preprocessed text
    """
    return text if not pipeline else preprocess_text(pipeline[0](text), pipeline[1:])


def text_preprocessor(pipeline: List) -> callable:
    """
    Return a function that preprocesses text using the given pipeline.

    :param pipeline: List of functions to be applied on the given text for preprocessing
    :return: Function that preprocesses text using the given pipeline
    """
    return lambda text: preprocess_text(text, pipeline)


def get_stopword_set(stopwords_file: str) -> set:
    """
    Return the set of stopwords from the given stopwords file.

    :param stopwords_file: Path to the file containing stopwords
    :return: Set of stopwords
    """
    with codecs.open(stopwords_file, 'r', 'utf-8') as f:
        return set([stopword.strip() for stopword in f])


def create_standard_text_preprocessor(stopwords_file: str) -> callable:
    """
    Return a standard text preprocessor using the given stopwords file.

    The text preprocessor performs the following steps:
    - removing special characters
    - removing numerals
    - converting all alphabets to lower case
    - removing stopwords using the stopword set obtained from the given stopwords file
    - stemming the words using Porter stemmer

    :param stopwords_file: Path to the file containing stopwords
    :return: Function that preprocesses text according to the pipeline
    """
    stopword_set = get_stopword_set(stopwords_file)

    # pipeline
    pipeline = [remove_special_characters,
                remove_numerals,
                convert_to_lower_case,
                lambda s: remove_stopwords(s, stopword_set),
                lambda s: ' '.join([stem_word(stemmed_token, snowballstemmer.stemmer('porter')) for stemmed_token in tokenize(s)])
                ]

    return text_preprocessor(pipeline)


def standard_text_preprocessor_1() -> callable:
    """
    Return a standard text preprocessor using the NLTK stopword list.

    This text preprocessor performs the same steps as the one created by the
    `create_standard_text_preprocessor` function using the NLTK stopword list.

    :return: Function that preprocesses text according to the pipeline
    """
    stopwords_file = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
    return create_standard_text_preprocessor(stopwords_file)


def standard_text_preprocessor_2() -> callable:
    """
    Return a standard text preprocessor using a custom stopword list.

    This text preprocessor performs the same steps as the one created by the
    `create_standard_text_preprocessor` function using a custom stopword list
    that excludes negation terms.

    :return: Function that preprocesses text according to the pipeline
    """
    stopwords_file = os.path.join(os.path.dirname(__file__), 'nonneg_stopwords.txt')
    return create_standard_text_preprocessor(stopwords_file)