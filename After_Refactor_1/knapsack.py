from collections import defaultdict
from typing import List, Tuple


def knapsack(capacity: int, items: List[Tuple[int, int]]) -> int:
    """
    Given a knapsack with a certain capacity and a list of items, returns the maximum value that can be packed into the knapsack.
    Each item has a weight and a value.
    """
    memo = defaultdict(int)

    for i in range(1, len(items) + 1):
        weight, value = items[i - 1]
        memo = _fill_knapsack(memo, weight, value, capacity)

    return memo[len(items), capacity]


def _fill_knapsack(
    memo: defaultdict,
    weight: int,
    value: int,
    capacity: int
) -> defaultdict:
    """
    Fills the knapsack with the given item by maximizing its value.
    """
    for j in range(1, capacity + 1):
        memo[i, j] = memo[i - 1, j]

        if weight <= j:
            memo[i, j] = max(
                memo[i, j],
                value + memo[i - 1, j - weight]
            )

    return memo
