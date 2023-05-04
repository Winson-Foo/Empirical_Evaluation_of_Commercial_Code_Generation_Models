def flatten_list(nested_list):
    """
    Flattens a nested list into a flat list by recursively iterating over the nested list
    """
    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            flat_list.extend(flatten_list(element))
        else:
            flat_list.append(element)
    return flat_list