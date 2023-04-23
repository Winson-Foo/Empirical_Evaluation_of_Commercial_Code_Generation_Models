def find_next_permutation(perm):
    """
    Given a permutation 'perm', find the next permutation in lexographic order and return it.
    If no next permutation exists, return None.
    """
    i = find_largest_i(perm)
    if i == -1:
        return None
    j = find_largest_j(perm, i)
    next_perm = swap(perm, i, j)
    reverse_suffix(next_perm, i+1)
    return next_perm
    
def find_largest_i(perm):
    """
    Find the largest index 'i' such that perm[i] < perm[i+1].
    If no such index exists, return -1.
    """
    for i in range(len(perm)-2, -1, -1):
        if perm[i] < perm[i+1]:
            return i
    return -1
    
def find_largest_j(perm, i):
    """
    Find the largest index 'j' such that perm[j] > perm[i].
    """
    for j in range(len(perm)-1, i, -1):
        if perm[j] > perm[i]:
            return j
            
def swap(perm, i, j):
    """
    Swap the elements at indices 'i' and 'j' in the list 'perm' and return the modified list.
    """
    next_perm = list(perm)
    next_perm[i], next_perm[j] = next_perm[j], next_perm[i]
    return next_perm
    
def reverse_suffix(perm, start):
    """
    Reverse the suffix of the list 'perm' starting from index 'start'.
    """
    perm[start:] = reversed(perm[start:])