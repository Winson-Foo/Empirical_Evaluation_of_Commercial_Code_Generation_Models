# Given a set, returns its powerset
def get_powerset(array):
    if array:
        # take the first element of the set and the rest of the elements as two separate lists
        first_element, *rest_elements = array
        # get the subsets of the rest of the elements recursively
        rest_subsets = get_powerset(rest_elements)
        # concatenate the subsets of the rest of the elements with the first element included 
        # to get the subsets of the entire set 
        return [[first_element] + subset for subset in rest_subsets] + rest_subsets
    else:
        # if the set is empty, return the set with an empty list as its only subset
        return [[]]