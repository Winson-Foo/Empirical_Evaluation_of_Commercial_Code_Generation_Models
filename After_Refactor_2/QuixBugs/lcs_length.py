from collections import Counter

def longest_common_subsequence_length(s: str, t: str) -> int:
    """
    Returns the length of the longest common subsequence of two strings.

    :param s: First input string.
    :param t: Second input string.
    :return: Length of longest common subsequence.
    """
    dp = [[0] * len(t) for _ in range(len(s))]

    def update_dp(i: int, j: int) -> None:
        if s[i] == t[j]:
            dp[i][j] = dp[i-1][j-1] + 1
        else:
            dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    for i, char_s in enumerate(s):
        for j, char_t in enumerate(t):
            update_dp(i, j)

    return dp[-1][-1]