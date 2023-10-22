import numba as nb

@nb.njit
def compute_set_edits_1(word):
    # generate all possible splits of the word
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    
    # generate all possible deletes by removing one character from the word
    deletes = [L + R[1:] for L, R in splits if R]
    
    # generate all possible transpositions by swapping adjacent characters in the word
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    
    # generate all possible replacements by replacing each character in the word with another letter
    letters = 'abcdefghijklmnopqrstuvwxyz'
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    
    # generate all possible insertions by adding each letter to each position in the word
    inserts = [L + c + R for L, R in splits for c in letters]

    # combine all sets of edits into one set and return it
    all_edits = set(deletes + transposes + replaces + inserts)
    return all_edits

@nb.njit
def compute_set_edits_2(word):
    # iterate through each first-level edit and generate all possible second-level edits
    second_level_edits = (e2 for e1 in compute_set_edits_1(word) for e2 in compute_set_edits_1(e1))
    return second_level_edits