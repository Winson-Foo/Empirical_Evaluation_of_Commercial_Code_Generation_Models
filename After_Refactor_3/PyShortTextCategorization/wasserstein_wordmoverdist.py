import warnings
from itertools import product

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix

from shorttext.utils.gensim_corpora import tokens_to_fracdict
from gensim.models.keyedvectors import KeyedVectors

def calculate_word_mover_distance_linprog(word_list_1: list, word_list_2: list, wvmodel: KeyedVectors, distancefunc=euclidean) -> np.ndarray:
    """ Compute the Word Mover's distance (WMD) between the two given lists of tokens, and return the LP problem class.

    Using methods of linear programming, supported by PuLP, calculate the WMD between two lists of words. A word-embedding
    model has to be provided. The whole `scipy.optimize.Optimize` object is returned.

    Reference: Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger, "From Word Embeddings to Document Distances," *ICML* (2015).

    Args:
    first_sent_tokens (list): first list of tokens.
    second_sent_tokens (list): second list of tokens.
    wvmodel (gensim.models.keyedvectors.KeyedVectors): word-embedding models.
    distancefunc (function): distance function that takes two numpy ndarray.

    Returns:
    Whole result of the linear programming problem(numpy.ndarray)
    """

    nb_tokens_1 = len(word_list_1)
    nb_tokens_2 = len(word_list_2)

    all_tokens = list(set(word_list_1 + word_list_2))
    word_vec_dict = {token: wvmodel[token] for token in all_tokens}

    word_list_1_buckets = tokens_to_fracdict(word_list_1)
    word_list_2_buckets = tokens_to_fracdict(word_list_2)

    collapsed_idx_func = lambda i, j: i * nb_tokens_2 + j

    # Assigning T
    T = np.zeros(nb_tokens_1 * nb_tokens_2)
    for i, j in product(range(nb_tokens_1), range(nb_tokens_2)):
        T[collapsed_idx_func(i, j)] = distancefunc(word_vec_dict[word_list_1[i]],
                                                   word_vec_dict[word_list_2[j]])

    # Assigning Aeq and beq
    Aeq = csr_matrix(
        (nb_tokens_1 + nb_tokens_2, nb_tokens_1 * nb_tokens_2)
    )
    beq = np.zeros(nb_tokens_1 + nb_tokens_2)
    for i in range(nb_tokens_1):
        for j in range(nb_tokens_2):
            Aeq[i, collapsed_idx_func(i, j)] = 1.
        beq[i] = word_list_1_buckets[word_list_1[i]]
    for j in range(nb_tokens_2):
        for i in range(nb_tokens_1):
            Aeq[j + nb_tokens_1, collapsed_idx_func(i, j)] = 1.
        beq[j + nb_tokens_1] = word_list_2_buckets[word_list_2[j]]

    return linprog(T, A_eq=Aeq, b_eq=beq)

def calculate_word_mover_distance(word_list_1: list, word_list_2: list, wvmodel: KeyedVectors, distancefunc=euclidean) -> float:
    """ Compute the Word Mover's distance (WMD) between the two given lists of tokens.

    Using methods of linear programming, calculate the WMD between two lists of words. A word-embedding
    model has to be provided. WMD is returned.

    Reference: Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger, "From Word Embeddings to Document Distances," *ICML* (2015).

    Args:
    first_sent_tokens (list): first list of tokens.
    second_sent_tokens (list): second list of tokens.
    wvmodel (gensim.models.keyedvectors.KeyedVectors): word-embedding models.
    distancefunc (function): distance function that takes two numpy ndarray.

    Returns:
    Word Mover's distance (WMD) (float)
    """

    linprog_result = calculate_word_mover_distance_linprog(word_list_1, word_list_2, wvmodel, distancefunc=distancefunc)

    return linprog_result['fun']