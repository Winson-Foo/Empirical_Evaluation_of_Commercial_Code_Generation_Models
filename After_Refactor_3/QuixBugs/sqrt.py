def calculate_square_root(x, epsilon):
    """
    Calculates the square root of x with a given degree of accuracy using the Newton-Raphson method.

    Parameters:
    x (float): The value whose square root is to be calculated
    epsilon (float): The maximum allowable error

    Returns:
    float: The calculated square root of x
    """
    guess = x / 2
    while abs(x - guess ** 2) > epsilon:
        guess = 0.5 * (guess + x / guess)
    return guess 