from itertools import product
from typing import List, Tuple

from .dldist import damerau_levenshtein
from .lcp import longest_common_prefix


def similarity(str1: str, str2: str) -> float:
    """ Return the similarity between the two strings.

    Return the similarity between the two strings, between 0 and 1 inclusively.
    The similarity is the maximum of the two values:
    - 1 - Damerau-Levenshtein distance between two strings / maximum length of the two strings
    - longest common prefix of the two strings / maximum length of the two strings

    Reference: Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen, 
    "Computer-Based Coding of Occupation Codes for Epidemiological Analyses," *2014 IEEE 27th 
    International Symposium on Computer-Based Medical Systems* (CBMS), pp. 347-350. (2014) 
    [`IEEE <http://ieeexplore.ieee.org/abstract/document/6881904/>`_]

    :param str1: a string
    :param str2: a string
    :return: similarity, between 0 and 1 inclusively
    :rtype: float
    """
    maxlen = max_length(str1, str2)
    editdistance = damerau_levenshtein(str1, str2)
    lcp = longest_common_prefix(str1, str2)
    return max(1. - float(editdistance)/maxlen, float(lcp)/maxlen)

def max_length(str1: str, str2: str) -> int:
    """ Return the maximum length of the two strings.

    :param str1: a string
    :param str2: a string
    :return: maximum length of the two strings
    :rtype: int
    """
    return max(len(str1), len(str2))

def combined_similarity(tokens1: List[str], tokens2: List[str]) -> List[Tuple[Tuple[str, str], float]]:
    """ Return a list of tuples of token pairs and their combined similarity scores.

    :param tokens1: list of tokens.
    :param tokens2: list of tokens.
    :return: list of tuples of token pairs and their combined similarity scores
    :rtype: list
    """
    sim_list = []
    for token1, token2 in product(tokens1, tokens2):
        sim = similarity(token1, token2)
        sim_list.append(((token1, token2), sim))
    sim_list.sort(key=lambda item: item[1], reverse=True)
    return sim_list

def soft_intersection_list(tokens1: List[str], tokens2: List[str]) -> List[Tuple[Tuple[str, str], float]]:
    """ Return the soft number of intersections between two lists of tokens.

    :param tokens1: list of tokens.
    :param tokens2: list of tokens.
    :return: soft number of intersections.
    :rtype: list
    """
    sim_list = combined_similarity(tokens1, tokens2)
    chosen_pairs = []
    used1, used2 = set(), set()
    for (token1, token2), sim in sim_list:
        if (token1 not in used1) and (token2 not in used2):
            chosen_pairs.append(((token1, token2), sim))
            used1.add(token1)
            used2.add(token2)
    return chosen_pairs

def numerator(tokens1: List[str], tokens2: List[str]) -> float:
    """ Return the numerator of the soft Jaccard score.

    :param tokens1: list of tokens.
    :param tokens2: list of tokens.
    :return: numerator of the soft Jaccard score.
    :rtype: float
    """
    return sum([item[1] for item in soft_intersection_list(tokens1, tokens2)])

def denominator(tokens1: List[str], tokens2: List[str]) -> float:
    """ Return the denominator of the soft Jaccard score.

    :param tokens1: list of tokens.
    :param tokens2: list of tokens.
    :return: denominator of the soft Jaccard score.
    :rtype: float
    """
    num_intersections = numerator(tokens1, tokens2)
    num_unions = len(tokens1) + len(tokens2) - num_intersections
    return float(num_unions)

def soft_jaccard_score(tokens1: List[str], tokens2: List[str]) -> float:
    """ Return the soft Jaccard score of the two lists of tokens, between 0 and 1 inclusively.

    Reference: Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen, 
    "Computer-Based Coding of Occupation Codes for Epidemiological Analyses," *2014 IEEE 27th 
    International Symposium on Computer-Based Medical Systems* (CBMS), pp. 347-350. (2014) 
    [`IEEE <http://ieeexplore.ieee.org/abstract/document/6881904/>`_]

    :param tokens1: list of tokens.
    :param tokens2: list of tokens.
    :return: soft Jaccard score, between 0 and 1 inclusively.
    :rtype: float
    """
    return numerator(tokens1, tokens2) / denominator(tokens1, tokens2) if denominator(tokens1, tokens2) > 0 else 0.0