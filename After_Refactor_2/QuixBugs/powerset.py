def powerset(arr):
    """
    Returns the powerset of an array
    :param arr: input array
    :return: powerset of the array
    """
    if arr:
        # get the first element and the rest of the array
        first, *rest = arr
        # get all the subsets of the rest of the array
        rest_subsets = powerset(rest)
        # combine the subsets of the rest of the array with the subsets that include the first element
        return [[first] + subset for subset in rest_subsets] + rest_subsets
    else:
        # empty array has only one subset, the empty array
        return [[]] 