package com.facebook.react.util;

import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableType;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class JSStackTrace {
    private static final String LINE_NUMBER_KEY = "lineNumber";
    private static final String FILE_KEY = "file";
    private static final String COLUMN_KEY = "column";
    private static final String METHOD_NAME_KEY = "methodName";
    
    private static final String FILE_ID_REGEX = "\\b((?:seg-\\d+(?:_\\d+)?|\\d+)\\.js)";

    /**
     * Formats a stack trace into a readable string.
     *
     * @param message the error message
     * @param stack the stack trace
     * @return a formatted string containing the error message and stack trace
     */
    public static String format(String message, ReadableArray stack) {
        StringBuilder sb = new StringBuilder(message).append(", stack:\n");
        for (int i = 0; i < stack.size(); i++) {
            ReadableMap frame = stack.getMap(i);
            sb.append(frame.getString(METHOD_NAME_KEY)).append("@").append(parseFileId(frame));

            if (frame.hasKey(LINE_NUMBER_KEY) && !frame.isNull(LINE_NUMBER_KEY) &&
                    frame.getType(LINE_NUMBER_KEY) == ReadableType.Number) {
                sb.append(frame.getInt(LINE_NUMBER_KEY));
            } else {
                sb.append(-1);
            }

            if (frame.hasKey(COLUMN_KEY) && !frame.isNull(COLUMN_KEY) &&
                    frame.getType(COLUMN_KEY) == ReadableType.Number) {
                sb.append(":").append(frame.getInt(COLUMN_KEY));
            }

            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * Extracts the file identifier from a stack frame.
     *
     * @param frame the stack frame
     * @return the file identifier
     */
    private static String parseFileId(ReadableMap frame) {
        if (frame.hasKey(FILE_KEY) && !frame.isNull(FILE_KEY) &&
                frame.getType(FILE_KEY) == ReadableType.String) {
            String file = frame.getString(FILE_KEY);
            if (file != null) {
                final Matcher matcher = Pattern.compile(FILE_ID_REGEX).matcher(file);
                if (matcher.find()) {
                    return matcher.group(1) + ":";
                }
            }
        }
        return "";
    }
}