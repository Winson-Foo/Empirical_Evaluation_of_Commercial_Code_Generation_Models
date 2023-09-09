import numba as nb
from typing import List, Set

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


@nb.njit
def compute_splits(word: str) -> List:
    """
    Splits a word into all possible combinations of left and right substrings.

    Args:
        word: A string representing the input word.

    Returns:
        A list of tuples representing all possible left and right substrings.
    """
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]


@nb.njit
def compute_deletes(splits: List) -> List:
    """
    Deletes a single character from every right substring in the input.

    Args:
        splits: A list of tuples representing all possible left and right substrings.

    Returns:
        A list of strings with a single character deleted from every right substring in the input.
    """
    return [L + R[1:] for L, R in splits if R]


@nb.njit
def compute_transposes(splits: List) -> List:
    """
    Transposes every adjacent character in every right substring in the input.

    Args:
        splits: A list of tuples representing all possible left and right substrings.

    Returns:
        A list of strings with every adjacent character transposed in every right substring in the input.
    """
    return [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]


@nb.njit
def compute_replaces(splits: List) -> List:
    """
    Replaces every character in every right substring in the input with every letter in the alphabet.

    Args:
        splits: A list of tuples representing all possible left and right substrings.

    Returns:
        A list of strings with every character in every right substring replaced with every letter in the alphabet.
    """
    return [L + c + R[1:] for L, R in splits if R for c in ALPHABET]


@nb.njit
def compute_inserts(splits: List) -> List:
    """
    Inserts every letter in the alphabet at every possible position in every right substring in the input.

    Args:
        splits: A list of tuples representing all possible left and right substrings.

    Returns:
        A list of strings with every letter in the alphabet inserted at every possible position in every right substring in the input.
    """
    return [L + c + R for L, R in splits for c in ALPHABET]


@nb.njit
def compute_set_edits1(word: str) -> Set:
    """
    Computes a set of all possible edits with distance 1 from the input word.

    Args:
        word: A string representing the input word.

    Returns:
        A set of all possible edits with distance 1 from the input word.
    """
    splits = compute_splits(word)
    deletes = compute_deletes(splits)
    transposes = compute_transposes(splits)
    replaces = compute_replaces(splits)
    inserts = compute_inserts(splits)

    return set(deletes + transposes + replaces + inserts)


@nb.njit
def compute_set_edits2(word: str) -> Set:
    """
    Computes a set of all possible edits with distance 2 from the input word.

    Args:
        word: A string representing the input word.

    Returns:
        A set of all possible edits with distance 2 from the input word.
    """
    return set(e2 for e1 in compute_set_edits1(word) for e2 in compute_set_edits1(e1))