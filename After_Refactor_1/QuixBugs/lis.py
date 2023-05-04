from typing import List

def lis(nums: List[int]) -> int:
    """
    Returns the length of the longest increasing subsequence in the given list of integers.
    """
    ends = {}
    longest_subsequence_length = 0

    for i, current_num in enumerate(nums):

        possible_prefix_lengths = [j for j in range(1, longest_subsequence_length + 1) if nums[ends[j]] < current_num]
        
        longest_prefix_length = max(possible_prefix_lengths) if possible_prefix_lengths else 0
        
        if longest_prefix_length == longest_subsequence_length or current_num < nums[ends[longest_prefix_length + 1]]:
            ends[longest_prefix_length + 1] = i
            longest_subsequence_length = max(longest_subsequence_length, longest_prefix_length + 1)

    return longest_subsequence_length