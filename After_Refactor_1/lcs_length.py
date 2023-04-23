from typing import List, Tuple

def lcs_length(s: str, t: str) -> int:
    dp = {}

    def _lcs_helper(s1: str, s2: str, i: int, j: int) -> int:
        if (i, j) in dp:
            return dp[(i, j)]
        if i == len(s1) or j == len(s2):
            return 0
        if s1[i] == s2[j]:
            dp[(i, j)] = _lcs_helper(s1, s2, i + 1, j + 1) + 1
        else:
            dp[(i, j)] = max(_lcs_helper(s1, s2, i, j + 1), _lcs_helper(s1, s2, i + 1, j))
        return dp[(i, j)]

    return _lcs_helper(s, t, 0, 0)