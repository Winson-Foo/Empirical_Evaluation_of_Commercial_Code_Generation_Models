from collections.abc import Iterable
from typing import List, Dict

def flatten(items: Iterable) -> List:
    """Returns flattened iterable from any nested iterable"""
    flattened_list = []
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                flattened_list.append(sub_x)
        else:
            flattened_list.append(x)
    return flattened_list

def has_cycle(producers: List) -> bool:
    '''
    Used to detect if the graph is not acyclical.  Returns true if it \
    finds a cycle in the graph.  It begins exploring the graph from producers down \
    all the way to consumers.
    '''
    visited = {}  # type: Dict
    rec = {}  # type: Dict
    for v in producers:
        visited[v] = False
        rec[v] = False
    
    for v in producers:
        if visited[v] == False:
            if _has_cycle_util(v, visited, rec):
                return True
    return False

def _has_cycle_util(v, visited, rec):
    '''
    - Arguments:
        - v : (Node)
        - visited (dict : Node -> boolean)
        - rec (dict : Node -> boolean)
    '''
    visited[v] = True
    rec[v] = True
    
    for child in v.children:
        if child not in visited:
            visited[child] = False
        if visited[child] == False:
            if _has_cycle_util(child, visited, rec):
                return True
        elif rec[child] == True:
            return True
    
    rec[v] = False
    return False

def topological_sort(producers: List) -> List:
    '''
    Creates a topological sort of the computation graph.

    - Arguments:
        - producers: a list of producer nodes, that is, nodes with no parents.
    
    - Returns:
        - stack: a list of nodes in topological order.  If \
            a *node A* appears before a *node B* on the list, it means \
            that *node A* does not depend on *node B* output
    '''
    visited = {}  # type: Dict
    for v in producers:
        visited[v] = False
    stack = []

    for v in producers:
        if visited[v] == False:
            _topological_sort_util(v, visited, stack)
    
    return stack

def _topological_sort_util(v, visited, stack):
    '''
    - Arguments:
        - v : (Node)
        - visited : (dict: node -> boolean)
        - stack: (list)
    '''
    visited[v] = True
    for child in v.children:
        if child not in visited or visited[child] == False:
            _topological_sort_util(child, visited, stack)
    stack.insert(0, v)