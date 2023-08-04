// To improve the maintainability of the code, you can make the following changes:

// 1. Add meaningful comments to the code to explain its functionality and logic.

// 2. Use descriptive variable names to make the code more understandable.

// 3. Use try-catch blocks to handle potential exceptions.

// 4. Use proper indentation and spacing to enhance code readability.

// 5. Remove unnecessary imports and unused code.

// Here is the refactored code:

package java_programs;

import java.util.ArrayList;

public class TextWrapper {
    public static void main(String[] args) {
        System.out.println(wrapText("abc", "c", 30));
    }

    /**
     * Wraps the given text into lines of specified maximum width.
     * 
     * @param text  the text to be wrapped
     * @param delimiter  the delimiter to be used for line breaks
     * @param cols  the maximum width for each line
     * @return  an ArrayList of wrapped lines
     */
    public static ArrayList<String> wrapText(String text, String delimiter, int cols) {
        ArrayList<String> lines = new ArrayList<>();
        try {
            while (text.length() > cols) {
                int end = text.lastIndexOf(delimiter, cols); // off by one?
                if (end == -1) {
                    end = cols;
                }
                String line = text.substring(0, end);
                text = text.substring(end);
                lines.add(line);
            }
            
            // Add the remaining text as the last line
            lines.add(text);
        } catch (Exception e) {
            System.out.println("An error occurred while wrapping the text: " + e.getMessage());
        }

        return lines;
    }
}

