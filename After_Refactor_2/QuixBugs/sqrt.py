def approximate_square_root(number, tolerance):
    """
    Calculates the approximate square root of a number using the Babylonian method.
    
    Parameters:
    number (float): The number to calculate the square root of.
    tolerance (float): The maximum allowed difference between the actual and approximate square root.
    
    Returns:
    float: The approximate square root of the number.
    """
    approximation = number / 2
    
    while abs(number - approximation ** 2) > tolerance:
        approximation = 0.5 * (approximation + number / approximation)
    
    return approximation 