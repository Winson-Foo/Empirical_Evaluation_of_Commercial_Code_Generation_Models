def possible_change(coins, total):
    if total == 0:
        return 1
    
    if len(coins) == 0 or total < 0:
        return 0

    current_coin, *remaining_coins = coins
    return possible_change(coins, total - current_coin) + possible_change(remaining_coins, total) 