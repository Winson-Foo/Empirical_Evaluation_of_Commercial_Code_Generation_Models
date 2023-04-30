from collections.abc import Iterable
from typing import List

def flatten(items: Iterable) -> List:
    """
    Returns flattened iterable from any nested iterable
    """
    result = []
    for item in items:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            for subitem in flatten(item):
                result.append(subitem)
        else:
            result.append(item)
    return result

def has_cycle(producers: List) -> bool:
    """
    Used to detect if the graph is not acyclical. Returns true if it 
    finds a cycle in the graph. It begins exploring the graph from producers down 
    all the way to consumers.
    """
    visited = {}
    rec = {}

    for node in producers:
        visited[node] = False
        rec[node] = False

    def _has_cycle_util(node):
        visited[node] = True
        rec[node] = True

        for child in node.children:
            if not visited.get(child, False):
                visited[child] = False
            if visited[child] is False:
                if _has_cycle_util(child):
                    return True
            elif rec[child] is True:
                return True

        rec[node] = False
        return False

    for node in producers:
        if visited[node] is False:
            if _has_cycle_util(node):
                return True

    return False

def topological_sort(producers: List) -> List:
    """
    Creates a topological sort of the computation graph.
    """
    visited = {}
    for node in producers:
        visited[node] = False
    stack = []

    def _topological_sort_util(node):
        visited[node] = True
        for child in node.children:
            if not visited.get(child, False) or visited[child] is False:
                _topological_sort_util(child)
        stack.insert(0, node)

    for node in producers:
        if visited[node] is False:
            _topological_sort_util(node)

    return stack
