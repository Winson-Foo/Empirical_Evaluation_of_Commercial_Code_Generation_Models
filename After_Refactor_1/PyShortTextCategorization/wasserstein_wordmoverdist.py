import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
from shorttext.utils.gensim_corpora import tokens_to_fracdict


def word_mover_distance_linear_programming(first_tokens, second_tokens, word_embeddings_model, distance_function=euclidean):
    """
    Compute the Word Mover's distance (WMD) between two given lists of tokens, and return the LP problem class.

    Using methods of linear programming, supported by PuLP, calculate the WMD between two lists of words. 

    :param first_tokens: First list of tokens.
    :type first_tokens: list
    :param second_tokens: Second list of tokens.
    :type second_tokens: list
    :param word_embeddings_model: KeyedVectors object containing word embeddings.
    :type word_embeddings_model: gensim.models.keyedvectors.KeyedVectors
    :param distance_function: distance function that takes two numpy ndarray. (default: euclidean)
    :type distance_function: function
    :return: result of the linear programming problem
    :rtype: scipy.optimize.OptimizeResult
    """
    num_tokens_first = len(first_tokens)
    num_tokens_second = len(second_tokens)
    all_tokens = list(set(first_tokens+second_tokens))
    word_vectors_dict = {token: word_embeddings_model[token] for token in all_tokens}

    first_tokens_buckets = tokens_to_fracdict(first_tokens)
    second_tokens_buckets = tokens_to_fracdict(second_tokens)
    collapsed_idx_func = lambda i, j: i*num_tokens_second + j

    # assigning distance matrix T
    T = np.zeros(num_tokens_first*num_tokens_second)
    for i, j in product(range(num_tokens_first), range(num_tokens_second)):
        T[collapsed_idx_func(i, j)] = distance_function(word_vectors_dict[first_tokens[i]],
                                                        word_vectors_dict[second_tokens[j]])

    # assigning Aeq and beq
    Aeq = csr_matrix(
        (num_tokens_first+num_tokens_second,
         num_tokens_first*num_tokens_second)
    )
    beq = np.zeros(num_tokens_first+num_tokens_second)
    for i in range(num_tokens_first):
        for j in range(num_tokens_second):
            Aeq[i, collapsed_idx_func(i, j)] = 1.
        beq[i] = first_tokens_buckets[first_tokens[i]]
    for j in range(num_tokens_second):
        for i in range(num_tokens_first):
            Aeq[j+num_tokens_first, collapsed_idx_func(i, j)] = 1.
        beq[j+num_tokens_first] = second_tokens_buckets[second_tokens[j]]

    return linprog(T, A_eq=Aeq, b_eq=beq)


def word_mover_distance(first_tokens, second_tokens, word_embeddings_model, distance_function=euclidean):
    """
    Compute the Word Mover's distance (WMD) between two given lists of tokens.

    Using methods of linear programming, calculate the WMD between two lists of words. 

    :param first_tokens: First list of tokens.
    :type first_tokens: list
    :param second_tokens: Second list of tokens.
    :type second_tokens: list
    :param word_embeddings_model: KeyedVectors object containing word embeddings.
    :type word_embeddings_model: gensim.models.keyedvectors.KeyedVectors
    :param distance_function: distance function that takes two numpy ndarray. (default: euclidean)
    :type distance_function: function
    :return: Word Mover's distance (WMD)
    :rtype: float
    """
    linprog_result = word_mover_distance_linear_programming(first_tokens, second_tokens, word_embeddings_model,
                                                             distance_function=distance_function)
    return linprog_result['fun']