def possible_change(coins: List[int], total: int) -> int:
    """
    Returns the number of ways to make change for a given total using a given set of coins

    Args:
        coins (List[int]): A list of integers representing the denominations of coins available
        total (int): An integer representing the desired total for which change needs to be made

    Returns:
        int: An integer representing the number of possible ways to make change for the given total
    
    """
    
    # Base cases
    if total == 0:
        return 1
    if len(coins) == 0 or total < 0:
        return 0
    
    # Recursive cases
    first, *rest = coins
    return possible_change(coins, total - first) + possible_change(rest, total)