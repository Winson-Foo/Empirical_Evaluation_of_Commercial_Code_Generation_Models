def kth(arr, k):
    pivot = arr[0]
    below, above = [], []
    for x in arr:
        if x < pivot:
            below.append(x)
        elif x > pivot:
            above.append(x)

    num_less = len(below)
    num_lessoreq = len(arr) - len(above)

    if k < num_less:
        return kth(below, k)
    elif k >= num_lessoreq:
        return kth(above, k - num_lessoreq)
    else:
        return pivot 