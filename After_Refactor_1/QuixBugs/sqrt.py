def square_root(number: float, epsilon: float) -> float:
    """
    Calculate the square root of a given number using the Newton-Raphson method
    
    :param number: The number to find the square root of
    :param epsilon: The acceptable error margin
    :return: The approximate square root of the number
    """
    approx_guess = number / 2
    while abs(number - approx_guess ** 2) > epsilon:
        approx_guess = 0.5 * (approx_guess + number / approx_guess)
    return approx_guess