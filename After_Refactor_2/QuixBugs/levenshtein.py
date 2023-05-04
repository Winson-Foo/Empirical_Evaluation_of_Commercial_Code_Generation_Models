def levenshtein(source, target):
    memo = {}
    
    def recursive_levenshtein(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        
        if i == len(source) or j == len(target):
            result = len(source[i:]) + len(target[j:])
        elif source[i] == target[j]:
            result = recursive_levenshtein(i+1, j+1)
        else:
            sub_cost = recursive_levenshtein(i+1, j+1)
            add_cost = recursive_levenshtein(i, j+1)
            del_cost = recursive_levenshtein(i+1, j)
            result = 1 + min(sub_cost, add_cost, del_cost)
        
        memo[(i, j)] = result
        return result
    
    return recursive_levenshtein(0, 0)