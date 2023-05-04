from collections import defaultdict

def knapsack(capacity, items):
    memo = defaultdict(int)

    for i, (weight, value) in enumerate(items, start=1):
        for j in range(1, capacity + 1):
            memo[i, j] = memo[i - 1, j]
            if weight <= j:
                memo[i, j] = max(memo[i, j], value + memo[i - 1, j - weight])

    return memo[len(items), capacity]