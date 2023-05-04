from typing import List, Union

def flatten(arr: List[Union[int, List]]) -> List[int]:
    """
    Flattens a nested list of integers into a single list.

    Args:
        arr: A list of integers or nested lists of integers.

    Returns:
        A flattened list of integers.
    """
    flattened = []
    for x in arr:
        if isinstance(x, list):
            for y in flatten(x):
                flattened.append(y)
        else:
            flattened.append(x)
    return flattened 