from itertools import product

import numpy as np
from scipy.spatial.distance import cosine

from shorttext.utils import tokenize


def get_word_similarities(tokens1, tokens2, wvmodel, sim_words):
    """
    Compute the similarities between every pair of words in two sentences using a specified similarity function.

    :param tokens1: list of words in sentence 1
    :param tokens2: list of words in sentence 2
    :param wvmodel: word-embedding model
    :param sim_words: function for calculating the similarities between a pair of word vectors
    :return: dictionary containing the similarities between all pairs of words
    """
    sim_dict = {}
    for i, j in product(range(len(tokens1)), range(len(tokens2))):
        if tokens1[i] in wvmodel and tokens2[j] in wvmodel:
            sim_dict[(i, j)] = sim_words(wvmodel[tokens1[i]], wvmodel[tokens2[j]])
    return sim_dict


def get_allowable_words(tokens, wvmodel):
    """
    Filter the list of words to only those that are present in the word-embedding model.

    :param tokens: list of words
    :param wvmodel: word-embedding model
    :return: list of allowable words
    """
    return list(filter(lambda w: w in wvmodel, tokens))


def get_intersection(sim_dict, allowable1, allowable2):
    """
    Compute the soft Jaccard score between two sentences.

    :param sim_dict: dictionary of word similarities
    :param allowable1: list indicating which words from sentence 1 can be used for calculating the Jaccard score
    :param allowable2: list indicating which words from sentence 2 can be used for calculating the Jaccard score
    :return: soft Jaccard score
    """
    intersection = 0.0
    sim_dict_items = sorted(sim_dict.items(), key=lambda s: s[1], reverse=True)
    for idxtuple, sim in sim_dict_items:
        i, j = idxtuple
        if allowable1[i] and allowable2[j]:
            intersection += sim
            allowable1[i] = False
            allowable2[j] = False
    return intersection


def jaccardscore_sents(sent1, sent2, wvmodel, sim_words=lambda vec1, vec2: 1-cosine(vec1, vec2)):
    """
    Compute the Jaccard score between sentences based on their word similarities.

    :param sent1: first sentence
    :param sent2: second sentence
    :param wvmodel: word-embeding model
    :param sim_words: function for calculating the similarities between a pair of word vectors (default: cosine)
    :return: soft Jaccard score
    """
    # Tokenize the sentences and filter the words to only those present in the word-embedding model
    tokens1 = get_allowable_words(tokenize(sent1), wvmodel)
    tokens2 = get_allowable_words(tokenize(sent2), wvmodel)

    # Compute the similarities between all pairs of words in the two sentences
    sim_dict = get_word_similarities(tokens1, tokens2, wvmodel, sim_words)

    # Calculate the soft Jaccard score based on the similarities and allowable words
    allowable1 = [True] * len(tokens1)
    allowable2 = [True] * len(tokens2)
    intersection = get_intersection(sim_dict, allowable1, allowable2)
    union = len(tokens1) + len(tokens2) - intersection
    if union > 0:
        return intersection / union
    elif intersection == 0:
        return 1.
    else:
        return np.inf