"""
This function returns the length of the longest increasing subsequence of a given array.

@param arr: list of integers
@return: length of LIS
"""

def get_longest_increasing_subsequence(arr):
    
    # dictionary to store the last element of the subsequence of a particular length
    end_of_subs = {}
    # length of LIS
    longest = 0

    for i, val in enumerate(arr):
        # find all subsequence prefix lengths that end with val
        prefix_lengths = [j for j in range(1, longest + 1) if arr[end_of_subs[j]] < val]
        # length of the longest prefix of which val can be a part of
        length = max(prefix_lengths) if prefix_lengths else 0
        # update the end_of_subs dictionary if necessary
        if length == longest or val < arr[end_of_subs[length + 1]]:
            end_of_subs[length + 1] = i
            longest = max(longest, length + 1)

    return longest 