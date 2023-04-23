def calculate_square_root(x: float, epsilon: float) -> float:
    """
    Returns the square root of a given number
    :param x: the number to find the square root of
    :param epsilon: the level of accuracy required
    :return: the square root of the given number
    """
    approx = x / 2  # start with a rough estimate
    while abs(x - approx * approx) > epsilon:  # continue iterating until required level of accuracy achieved
        approx = 0.5 * (approx + x / approx)  # improve the estimate
    return approx