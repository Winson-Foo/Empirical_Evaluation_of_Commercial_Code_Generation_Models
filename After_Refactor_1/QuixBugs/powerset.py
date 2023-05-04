from typing import List

def powerset(arr: List[int]) -> List[List[int]]:
    """
    Returns the powerset of a given list.

    Args:
        arr: The list of integers to generate the powerset of.

    Returns:
        A list of all subsets of the input list.
    """
    if arr:
        first, *rest = arr
        rest_subsets = powerset(rest)
        return [[first] + subset for subset in rest_subsets] + rest_subsets
    else:
        return [[]]