from typing import List

def powerset(arr: List[int]) -> List[List[int]]:
    """
    Returns all the subsets of a list.

    Args:
        arr: A list of integers.

    Returns:
        A list of all the possible subsets of the input list.
    """
    if arr:
        first, *rest = arr
        rest_subsets = powerset(rest)
        return [[first] + subset for subset in rest_subsets] + rest_subsets
    else:
        return [[]] 