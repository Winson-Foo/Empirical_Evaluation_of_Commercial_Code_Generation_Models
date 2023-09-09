from itertools import product
from typing import List, Tuple

from .dldist import damerau_levenshtein
from .lcp import longest_common_prefix


def similarity(word1: str, word2: str) -> float:
    """Return the similarity between the two words.

    Args:
        word1: a word.
        word2: a word.

    Returns:
        similarity, between 0 and 1 inclusively

    References:
        Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen,
        "Computer-Based Coding of Occupation Codes for Epidemiological Analyses,"
        *2014 IEEE 27th International Symposium on Computer-Based Medical Systems*
        (CBMS), pp. 347-350. (2014) [`IEEE
        <http://ieeexplore.ieee.org/abstract/document/6881904/>`_]
    """
    maxlen = max(len(word1), len(word2))
    editdistance = damerau_levenshtein(word1, word2)
    lcp = longest_common_prefix(word1, word2)
    return max(1 - editdistance / maxlen, lcp / maxlen)


def soft_intersection_list(tokens1: List[str], tokens2: List[str]) -> List[Tuple[Tuple[str, str], float]]:
    """Return the soft number of intersections between two lists of tokens.

    Args:
        tokens1: list of tokens.
        tokens2: list of tokens.

    Returns:
        List of tuples with token tuples and their similarity score.

    """
    intersected_list = sorted(
        [((token1, token2), similarity(token1, token2))
         for token1, token2 in product(tokens1, tokens2)],
        key=lambda item: item[1], reverse=True
    )

    included_list = set()
    used_tokens1 = set()
    used_tokens2 = set()
    for (token1, token2), sim in intersected_list:
        if token1 not in used_tokens1 and token2 not in used_tokens2:
            included_list.add(((token1, token2), sim))
            used_tokens1.add(token1)
            used_tokens2.add(token2)

    return list(included_list)


def soft_jaccard_score(tokens1: List[str], tokens2: List[str]) -> float:
    """Return the soft Jaccard score of the two lists of tokens, between 0 and 1 inclusively.

    Args:
        tokens1: list of tokens.
        tokens2: list of tokens.

    Returns:
        float: a value between 0 and 1 inclusively representing the soft Jaccard score.

    References:
        Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen,
        "Computer-Based Coding of Occupation Codes for Epidemiological Analyses,"
        *2014 IEEE 27th International Symposium on Computer-Based Medical Systems*
        (CBMS), pp. 347-350. (2014) [`IEEE
        <http://ieeexplore.ieee.org/abstract/document/6881904/>`_]

    """
    intersection_list = soft_intersection_list(tokens1, tokens2)
    num_intersections = sum(item[1] for item in intersection_list)
    num_unions = len(tokens1) + len(tokens2) - num_intersections
    return float(num_intersections) / num_unions