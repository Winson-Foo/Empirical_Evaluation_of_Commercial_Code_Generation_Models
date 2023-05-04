def generate_pascal_triangle(num_rows):
    triangle = [[1]]
    for row in range(1, num_rows):
        new_row = []
        for col in range(0, row + 1):
            if col > row - 1:
                val = 0
            else:
                val = triangle[row - 1][col]
            if col > 0:
                val += triangle[row - 1][col - 1]
            new_row.append(val)
        triangle.append(new_row)
    return triangle