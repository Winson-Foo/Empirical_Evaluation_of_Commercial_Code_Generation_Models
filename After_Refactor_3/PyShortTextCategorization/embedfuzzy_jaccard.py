from itertools import product
from typing import List, Tuple
import numpy as np

from scipy.spatial.distance import cosine
from shorttext.utils import tokenize

WordVecModel = 'gensim.models.keyedvectors.KeyedVectors'


def jaccardscore_sents(sent1: str, sent2: str,
                        wvmodel: WordVecModel,
                        sim_words=lambda vec1, vec2: 1 - cosine(vec1, vec2)) -> float:
    """ Compute the Jaccard score between sentences based on their word similarities.

    :param sent1: first sentence
    :param sent2: second sentence
    :param wvmodel: word-embedding model
    :param sim_words: function for calculating the similarities between a pair of word vectors (default: cosine)
    :return: soft Jaccard score
    """
    tokens1, tokens2 = tokenize_text(sent1), tokenize_text(sent2)
    tokens1, tokens2 = filter_word_vectors(tokens1, tokens2, wvmodel)
    allowable1, allowable2 = initialise_allowable_tokens(tokens1), initialise_allowable_tokens(tokens2)

    simdict = calculate_similarity_dict(tokens1, tokens2, sim_words, wvmodel)
    intersection = calculate_intersection(simdict, allowable1, allowable2)

    union = calculate_union(tokens1, tokens2, intersection)
    return compute_jaccard_score(intersection, union)


def tokenize_text(sentence: str) -> List[str]:
    """ Tokenize a given sentence

    :param sentence: sentence to tokenize
    :return: list of tokens
    """
    return tokenize(sentence)


def filter_word_vectors(tokens1: List[str], tokens2: List[str],
                        wvmodel: WordVecModel) -> Tuple[List[str], List[str]]:
    """ Filter out words that do not have word vectors in the word embedding model

    :param tokens1: list of tokens of first sentence
    :param tokens2: list of tokens of second sentence
    :param wvmodel: word embedding model
    :return: filtered tokens
    """
    tokens1 = list(filter(lambda w: w in wvmodel, tokens1))
    tokens2 = list(filter(lambda w: w in wvmodel, tokens2))
    return tokens1, tokens2


def initialise_allowable_tokens(tokens: List[str]) -> List[bool]:
    """ Initialise a list of allowable tokens for a given set of tokens

    :param tokens: list of tokens
    :return: list of allowable tokens
    """
    return [True] * len(tokens)


def calculate_similarity_dict(tokens1: List[str], tokens2: List[str],
                              sim_words, wvmodel: WordVecModel) -> dict:
    """ Calculate the similarity between pairs of words in two given sets of tokens

    :param tokens1: list of tokens of first sentence
    :param tokens2: list of tokens of second sentence
    :param sim_words: function for calculating similarity between two word vectors
    :param wvmodel: word embedding model
    :return: dictionary containing similarity scores
    """
    simdict = {(i, j): sim_words(wvmodel[tokens1[i]], wvmodel[tokens2[j]])
               for i, j in product(range(len(tokens1)), range(len(tokens2)))}
    return simdict


def calculate_intersection(simdict: dict, allowable1: List[bool], allowable2: List[bool]) -> float:
    """ Calculate the intersection between two sets of tokens based on their similarities

    :param simdict: dictionary containing similarity scores
    :param allowable1: list of allowable tokens in first sentence
    :param allowable2: list of allowable tokens in second sentence
    :return: intersection between two sets of tokens
    """
    intersection = 0.0
    simdictitems = sorted(simdict.items(), key=lambda s: s[1], reverse=True)
    for idxtuple, sim in simdictitems:
        i, j = idxtuple
        if allowable1[i] and allowable2[j]:
            intersection += sim
            allowable1[i] = False
            allowable2[j] = False
    return intersection


def calculate_union(tokens1: List[str], tokens2: List[str], intersection: float) -> float:
    """ Calculate the union between two sets of tokens

    :param tokens1: list of tokens of first sentence
    :param tokens2: list of tokens of second sentence
    :param intersection: intersection between two sets of tokens
    :return: union between two sets of tokens
    """
    return len(tokens1) + len(tokens2) - intersection


def compute_jaccard_score(intersection: float, union: float) -> float:
    """ Compute the Jaccard score between two sets of tokens

    :param intersection: intersection between two sets of tokens
    :param union: union between two sets of tokens
    :return: Jaccard score
    """
    if union > 0:
        return intersection / union
    elif intersection == 0:
        return 1.
    else:
        return np.inf