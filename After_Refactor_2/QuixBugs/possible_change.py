def calculate_coin_combinations(coin_values, target_total):
    if target_total == 0:
        return 1
    if not coin_values:
        return 0
    first_coin, *rest_coins = coin_values
    if target_total < 0:
        return 0
    return (calculate_coin_combinations(coin_values, target_total - first_coin) +
            calculate_coin_combinations(rest_coins, target_total)) 