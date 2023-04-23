def wrap(text, cols):
    lines = []
    while len(text) > cols:
        end = find_last_space(text, cols)
        line, text = text[:end], text[end:]
        lines.append(line)

    lines.append(text)
    return lines

def find_last_space(text, cols):
    # Helper function to find the last space before the given column
    end = text.rfind(' ', 0, cols + 1)
    if end == -1:
        return cols
    return end