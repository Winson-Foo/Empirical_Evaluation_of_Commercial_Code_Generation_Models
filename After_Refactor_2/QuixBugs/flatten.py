def flatten(arr):
    """
    This function flattens a nested list.
    """
    flat_list = []
    for item in arr:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list