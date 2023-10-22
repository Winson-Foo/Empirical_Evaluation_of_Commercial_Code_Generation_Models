from collections.abc import Iterable

def flatten(items):
    """
    Returns flattened iterable from any nested iterable
    """
    to_return = []
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            to_return.extend(flatten(x))
        else:
            to_return.append(x)
    return to_return


def has_cycle(producers):
    """
    Used to detect if the graph is not acyclical.
    Returns True if it finds a cycle in the graph.
    It begins exploring the graph from producers down to consumers.
    """
    visited = {}
    rec = {}
    for v in producers:
        visited[v] = False
        rec[v] = False

    def _has_cycle_util(v):
        nonlocal visited, rec
        visited[v] = True
        rec[v] = True

        for child in v.children:
            if child not in visited:
                visited[child] = False
            if not visited[child]:
                if _has_cycle_util(child):
                    return True
            elif rec[child]:
                return True

        rec[v] = False
        return False

    for v in producers:
        if not visited[v]:
            if _has_cycle_util(v):
                return True
    return False


def topological_sort(producers):
    """
    Creates a topological sort of the computation graph.

    Arguments:
    - producers: A list of producer nodes, that is, nodes with no parents.

    Returns:
    - stack: A list of nodes in topological order. If a node A appears before a node B on the list, it means that
    node A does not depend on node B output.
    """
    visited = {}
    for v in producers:
        visited[v] = False
    stack = []

    def _topological_sort_util(v):
        nonlocal visited, stack
        visited[v] = True
        for child in v.children:
            if child not in visited or not visited[child]:
                _topological_sort_util(child)
        stack.insert(0, v)

    for v in producers:
        if not visited[v]:
            _topological_sort_util(v)

    return stack