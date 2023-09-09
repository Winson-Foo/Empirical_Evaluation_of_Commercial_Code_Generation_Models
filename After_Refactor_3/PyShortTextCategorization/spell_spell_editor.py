import numba as nb

def get_splits(word):
    """
    Returns a list of all possible splits of a string.
    """
    return [(word[:i], word[i:]) for i in range(len(word) + 1)]

def get_deletes(splits):
    """
    Returns a list of strings formed by deleting one letter from each split.
    """
    return [L + R[1:] for L, R in splits if R]

def get_transposes(splits):
    """
    Returns a list of strings formed by swapping adjacent letters in each split.
    """
    return [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

def get_replaces(splits, letters):
    """
    Returns a list of strings formed by replacing one letter in each split with each letter in the alphabet.
    """
    return [L + c + R[1:] for L, R in splits if R for c in letters]

def get_inserts(splits, letters):
    """
    Returns a list of strings formed by inserting each letter in the alphabet into each possible position of each split.
    """
    return [L + c + R for L, R in splits for c in letters]

@nb.njit
def compute_set_edits1(word):
    """
    Returns a set of all possible strings that can be formed by deleting, transposing, replacing, or inserting one letter in the input word.
    """
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = get_splits(word)
    deletes = get_deletes(splits)
    transposes = get_transposes(splits)
    replaces = get_replaces(splits, letters)
    inserts = get_inserts(splits, letters)
    returned_set = set(deletes + transposes + replaces + inserts)
    return returned_set

@nb.njit
def compute_set_edits2(word):
    """
    Returns a generator that yields all possible strings that can be formed by applying compute_set_edits1() twice.
    """
    return (e2 for e1 in compute_set_edits1(word) for e2 in compute_set_edits1(e1))

# Example usage:
word = 'hello'
edits = compute_set_edits1(word)
print(edits)
bigger_edits = compute_set_edits2(word)
print(list(bigger_edits))