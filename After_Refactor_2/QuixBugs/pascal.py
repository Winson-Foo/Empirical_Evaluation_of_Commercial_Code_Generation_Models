def pascal_triangle(n):
    """
    Returns Pascal's triangle of n rows in the form of a list of lists.
    """
    triangle = [[1]]
    for row in range(1, n):
        current_row = []
        for column in range(0, row + 1):
            upleft = triangle[row - 1][column - 1] if column > 0 else 0
            upright = triangle[row - 1][column] if column < row else 0
            current_row.append(upleft + upright)
        triangle.append(current_row)

    return triangle 