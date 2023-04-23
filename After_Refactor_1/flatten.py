def flatten_list(nested_list: List[Any]) -> List[Any]:
    flat_list = []

    if not nested_list:
        return flat_list

    for x in nested_list:
        if isinstance(x, list):
            flat_list.extend(flatten_list(x))
        else:
            flat_list.append(x)

    return flat_list