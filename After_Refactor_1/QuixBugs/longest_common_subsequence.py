def longest_common_subsequence(a, b):
    if not a or not b:
        return ''

    first_char_a = a[0]
    first_char_b = b[0]

    if first_char_a == first_char_b:
        return first_char_a + longest_common_subsequence(a[1:], b[1:])
    else:
        lcs1 = longest_common_subsequence(a, b[1:])
        lcs2 = longest_common_subsequence(a[1:], b)

        return max(lcs1, lcs2, key=len)