from typing import Dict

def parse_label_map(path_to_labels: str) -> Dict[int, str]:
    '''
    - Arguments:
        - path_to_labels (str): path to pbtx file

    - Returns:
        - dict of form { id(int) : label(str)}
    '''
    with open(path_to_labels, "r") as f:
        text = f.read()
    
    entries = text.split('item')
    entry_pairs = []
    
    for entry in entries[1:]:
        index = int(entry.split('id:')[1].split('\n')[0])
        
        klass_name = entry.split('display_name:')[1].split('\'')[1]
        
        entry_pairs.append((index, klass_name))
        
    return dict(entry_pairs)
