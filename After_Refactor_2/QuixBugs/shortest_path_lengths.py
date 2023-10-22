from collections import defaultdict

def calculate_shortest_path_lengths(n, length_by_edge):
    lengths = defaultdict(lambda: float('inf'))
    for i in range(n):
        lengths[(i, i)] = 0
    lengths.update(length_by_edge)

    for k in range(n):
        for i in range(n):
            for j in range(n):
                lengths[(i, j)] = min(
                    lengths[(i, j)],
                    lengths[(i, k)] + lengths[(k, j)]
                )

    return lengths 