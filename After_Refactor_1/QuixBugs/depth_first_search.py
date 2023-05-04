from typing import Any, List


def is_reachable(start_node: Any, goal_node: Any) -> bool:
    visited_nodes = set()

    def has_visited(node: Any) -> bool:
        return node in visited_nodes

    def search_from(node: Any) -> bool:
        if has_visited(node):
            return False
        if node is goal_node:
            return True
        visited_nodes.add(node)
        return any(search_from(next_node) for next_node in node.successors)

    return search_from(start_node)