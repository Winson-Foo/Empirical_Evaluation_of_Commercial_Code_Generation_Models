def get_upleft(rows, r, c):
    if c > 0:
        return rows[r - 1][c - 1]
    else:
        return 0

def get_upright(rows, r, c):
    if c < r:
        return rows[r - 1][c]
    else:
        return 0

def generate_pascal_triangle(n):
    triangle = [[1]]
    for row in range(1, n):
        new_row = []
        for col in range(0, row + 1):
            upleft = get_upleft(triangle, row, col)
            upright = get_upright(triangle, row, col)
            new_value = upleft + upright
            new_row.append(new_value)
        triangle.append(new_row)
    return triangle