from typing import List, Tuple

def parse_label_map(path_to_labels: str) -> dict:
    """
    Parse the label map file and return a dictionary of class id to class name mappings
    :param path_to_labels: str - path to the label map pbtxt file
    :return: dict - dictionary of form {class_id(int): class_name(str)}
    """
    with open(path_to_labels, "r") as file:
        label_map = file.read()

    mappings = []
    index_start = label_map.find('item')

    while index_start != -1:
        id_start = label_map.find('id:', index_start)
        id_end = label_map.find('\n', id_start)
        class_id = int(label_map[id_start + len("id:"): id_end])

        name_start = label_map.find('display_name:', index_start)
        name_end = max(label_map.find("\'", name_start), label_map.find('\"', name_start))
        name_end = max(label_map.find("\'", name_end + 1), label_map.find('\"', name_end + 1))
        class_name = label_map[name_start + 1: name_end]

        mappings.append((class_id, class_name))
        index_start = label_map.find('item', name_end)
    
    return dict(mappings)
