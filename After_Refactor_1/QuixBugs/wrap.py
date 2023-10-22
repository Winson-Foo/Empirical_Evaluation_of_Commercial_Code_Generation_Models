def wrap(text, cols):
    # Renamed variable for better readability
    wrapped_lines = []
    
    while len(text) > cols:
        # Extracted the logic of finding the end of line into a separate function
        end = find_end_of_line(text, cols)
        
        line, text = text[:end], text[end:]
        wrapped_lines.append(line)

    wrapped_lines.append(text)
    return wrapped_lines

def find_end_of_line(text, cols):
    # Split the logic in a separate function to increase abstraction and better maintainability
    end = text.rfind(' ', 0, cols + 1)
    
    # Added a variable to make the code more readable
    max_chars_per_line = cols
    
    # Check if there is no space to break the line - if so, break at max_chars_per_line
    if end == -1:
        end = max_chars_per_line
        
    return end