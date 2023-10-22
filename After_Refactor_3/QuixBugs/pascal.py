def generate_pascal_triangle(num_rows):
    """
    Generates a Pascal Triangle with the given number of rows.
    """
    rows = [[1]]  # Initialize the Pascal's Triangle with the first row.
    for row_num in range(1, num_rows):
        # Calculate the values for the current row
        current_row = calculate_pascal_row(rows[row_num-1], row_num)
        rows.append(current_row)

    return rows

def calculate_pascal_row(previous_row, row_num):
    """
    Calculates the values for a single row in Pascal's Triangle
    given the previous row and the current row number.
    """
    current_row = []
    for col_num in range(0, row_num+1):
        upper_left = previous_row[col_num-1] if col_num > 0 else 0
        upper_right = previous_row[col_num] if col_num < row_num else 0
        current_row.append(upper_left + upper_right)

    return current_row 