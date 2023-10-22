from itertools import product

import numpy as np
from scipy.spatial.distance import cosine
from shorttext.utils import tokenize

def filter_tokens(tokens, wvmodel):
    """Filters a list of word tokens to only include those present in the word-embedding model.

    :param tokens: list of word tokens
    :param wvmodel: word-embedding model
    :return: filtered list of tokens
    :rtype: list
    """
    return list(filter(lambda w: w in wvmodel, tokens))

def get_similarity(wvmodel, token1, token2):
    """Calculates the similarity between two word vectors in the word-embedding model using cosine similarity.

    :param wvmodel: word-embedding model
    :param token1: first word token
    :param token2: second word token
    :return: cosine similarity between the two word vectors
    :rtype: float
    """
    return 1 - cosine(wvmodel[token1], wvmodel[token2])

def get_similarity_matrix(tokens1, tokens2, wvmodel):
    """Calculates the pairwise similarity between all combinations of tokens in two lists using cosine similarity.

    :param tokens1: first list of word tokens
    :param tokens2: second list of word tokens
    :param wvmodel: word-embedding model
    :return: similarity matrix of shape (len(tokens1), len(tokens2))
    :rtype: numpy.ndarray
    """
    sim_matrix = np.zeros((len(tokens1), len(tokens2)))
    for i, j in product(range(len(tokens1)), range(len(tokens2))):
        sim_matrix[i, j] = get_similarity(wvmodel, tokens1[i], tokens2[j])
    return sim_matrix

def jaccard_similarity(sent1, sent2, wvmodel, sim_words=get_similarity):
    """Compute the Jaccard similarity between two sentences based on the similarities between their word tokens.

    :param sent1: first sentence
    :param sent2: second sentence
    :param wvmodel: word-embedding model
    :param sim_words: function for calculating the similarity between a pair of word tokens (default: cosine similarity)
    :return: Jaccard similarity score
    :rtype: float
    """
    tokens1 = tokenize(sent1)
    tokens2 = tokenize(sent2)
    tokens1 = filter_tokens(tokens1, wvmodel)
    tokens2 = filter_tokens(tokens2, wvmodel)
    sim_matrix = get_similarity_matrix(tokens1, tokens2, wvmodel)

    intersection = sim_matrix.max(axis=1).sum()
    union = len(tokens1) + len(tokens2) - intersection

    if union > 0:
        return intersection / union
    elif intersection == 0:
        return 1.
    else:
        raise ValueError("Intersection cannot be negative")