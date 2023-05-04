from collections import Counter

def lcs_length(s, t):
    dp = Counter()
    for i, s_char in enumerate(s):
        for j, t_char in enumerate(t):
            if s_char == t_char:
                dp[i, j] = dp[i - 1, j - 1] + 1
    return max(dp.values(), default=0) 