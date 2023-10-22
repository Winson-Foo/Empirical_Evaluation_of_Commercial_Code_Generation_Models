def wrap(text, cols):
    lines = []
    while len(text) > cols:
        end = find_wrap_index(text, cols)
        line, text = text[:end], text[end:]
        lines.append(line)

    lines.append(text)
    return lines


def find_wrap_index(text, cols):
    wrap_index = text.rfind(' ', 0, cols + 1)
    if wrap_index == -1:
        wrap_index = cols
    return wrap_index 