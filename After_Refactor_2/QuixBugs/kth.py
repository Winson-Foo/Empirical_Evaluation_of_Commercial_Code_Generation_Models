# Define a helper function to partition the array based on a pivot
def partition(arr, pivot):
    below = [x for x in arr if x < pivot]
    above = [x for x in arr if x > pivot]
    return below, above

# Define the main function to find the kth smallest element in the array
def kth(arr, k):
    if len(arr) == 1:
        return arr[0]
    
    pivot = arr[0]
    below, above = partition(arr[1:], pivot)
    num_less = len(below)
    num_lessoreq = len(arr) - len(above)

    if k < num_less:
        return kth(below, k)
    elif k >= num_lessoreq:
        return kth(above, k - num_lessoreq)
    else:
        return pivot