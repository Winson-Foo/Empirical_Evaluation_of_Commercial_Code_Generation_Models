def possible_change(coins_list, total_amount):
    """
    Returns the number of possible ways to make change for the given amount
    using the coins in the list.
    """
    if total_amount == 0:
        return 1
    
    if not coins_list:
        return 0
    
    if total_amount < 0:
        return 0
    
    first_coin, *rest_of_coins = coins_list
    
    return possible_change(coins_list, total_amount - first_coin) + possible_change(rest_of_coins, total_amount)