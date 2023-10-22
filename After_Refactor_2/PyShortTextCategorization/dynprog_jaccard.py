from itertools import product

from .dldist import damerau_levenshtein
from .lcp import longest_common_prefix


def similarity(word1, word2):
    """
    Determines the similarity between the two words, returned as a float between 0 and 1 inclusive.
    The similarity is calculated as the maximum of these values:
    - 1 - Damerau-Levenshtein distance between two words / maximum length of the two words
    - longest common prefix of the two words / maximum length of the two words

    Reference: Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen, "Computer-Based 
    Coding of Occupation Codes for Epidemiological Analyses," *2014 IEEE 27th International Symposium on 
    Computer-Based Medical Systems* (CBMS), pp. 347-350. (2014) [IEEE]

    :param word1: string
    :param word2: string
    :return: float between 0 and 1 inclusive
    """
    max_length = max(len(word1), len(word2))
    edit_distance = damerau_levenshtein(word1, word2)
    longest_common_prefix_value = longest_common_prefix(word1, word2)
    return max(1. - float(edit_distance) / max_length, float(longest_common_prefix_value) / max_length)


def soft_intersection_list(tokens_list_1, tokens_list_2):
    """
    Determines the soft number of intersections between two lists of tokens.

    :param tokens_list_1: list of tokens
    :param tokens_list_2: list of tokens
    :return: list of tuples, with the similarity value between the tokens 
             and a tuple with both tokens that generated the similarity value
    """
    list_of_similarities = []
    for token1, token2 in product(tokens_list_1, tokens_list_2):
        similarity_value = similarity(token1, token2)
        list_of_similarities.append(((token1, token2), similarity_value))

    sorted_list_of_similarities = sorted(list_of_similarities, key=lambda item: item[1], reverse=True)

    included_list = set()
    used_tokens_1 = set()
    used_tokens_2 = set()
    for (token1, token2), sim in sorted_list_of_similarities:
        if (not (token1 in used_tokens_1)) and (not (token2 in used_tokens_2)):
            included_list.add(((token1, token2), sim))
            used_tokens_1.add(token1)
            used_tokens_2.add(token2)

    return included_list


def soft_jaccard_score(tokens_list_1, tokens_list_2):
    """
    Determines the soft Jaccard score of the two lists of tokens, returned as a float between 0 and 1 inclusive.

    Reference: Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen, "Computer-Based 
    Coding of Occupation Codes for Epidemiological Analyses," *2014 IEEE 27th International Symposium on 
    Computer-Based Medical Systems* (CBMS), pp. 347-350. (2014) [IEEE]

    :param tokens_list_1: list of tokens
    :param tokens_list_2: list of tokens
    :return: float between 0 and 1 inclusive
    """
    intersection_list = soft_intersection_list(tokens_list_1, tokens_list_2)
    num_intersections = sum([item[1] for item in intersection_list])
    num_unions = len(tokens_list_1) + len(tokens_list_2) - num_intersections
    return float(num_intersections) / float(num_unions)