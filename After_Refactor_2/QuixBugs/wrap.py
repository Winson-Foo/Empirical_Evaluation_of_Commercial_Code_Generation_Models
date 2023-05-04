def wrap(text, cols):
    # Split the text into words
    words = text.split()
    # Initialize an empty list to store the lines
    lines = []
    # Initialize an empty string to store the current line
    current_line = ""
    # Loop through the words
    for word in words:
        # Check if adding the next word exceeds the column limit
        if len(current_line + word) > cols:
            # Add the current line to the list of lines
            lines.append(current_line)
            # Start a new line with the current word
            current_line = word + " "
        else:
            # Add the current word to the current line
            current_line += word + " "

    # Add the last line to the list of lines
    lines.append(current_line)
    # Return the list of lines
    return lines 