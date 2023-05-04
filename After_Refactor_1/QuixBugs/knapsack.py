def knapsack(capacity, items):
    memo = {}

    for i, (weight, value) in enumerate(items, start=1):
        for j in range(capacity + 1):
            if (i-1, j) not in memo:
                memo[i-1, j] = 0
            memo[i, j] = memo[i-1, j]
            if weight <= j:
                memo[i, j] = max(memo[i, j], value + memo[i-1, j-weight])

    return memo[len(items), capacity]