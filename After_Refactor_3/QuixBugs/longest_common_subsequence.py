def longest_common_subsequence(a, b):
    if not a or not b:
        # Return empty string if either a or b is empty
        return ''

    if a[0] == b[0]:
        # If the first characters match, include in LCS and check the rest of the strings
        return a[0] + longest_common_subsequence(a[1:], b[1:])

    # Else, find LCS by checking the rest of the strings
    lcs1 = longest_common_subsequence(a, b[1:])
    lcs2 = longest_common_subsequence(a[1:], b)

    # Return the LCS with the maximum length
    return max(lcs1, lcs2, key=len) 