package com.facebook.react.util;

import com.facebook.react.bridge.ReadableArray;
import com.facebook.react.bridge.ReadableMap;
import com.facebook.react.bridge.ReadableType;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class JSStackTrace {

    interface KEYS {
        String LINE_NUMBER = "lineNumber";
        String FILE = "file";
        String COLUMN = "column";
        String METHOD_NAME = "methodName";
    }

    private static final Pattern FILE_ID_PATTERN =
            Pattern.compile("\\b((?:seg-\\d+(?:_\\d+)?|\\d+)\\.js)");

    public static String format(String message, ReadableArray stack) {
        StringBuilder stringBuilder = new StringBuilder(message).append(", stack:\n");
        for (int i = 0; i < stack.size(); i++) {
            ReadableMap frame = stack.getMap(i);
            String methodName = frame.getString(KEYS.METHOD_NAME);
            String fileId = parseFileId(frame);
            String lineNumber = getNumberString(frame, KEYS.LINE_NUMBER);
            String columnNumber = getNumberString(frame, KEYS.COLUMN);

            stringBuilder.append(concatFrame(methodName, fileId, lineNumber, columnNumber));
        }
        return stringBuilder.toString();
    }

    private static String getNumberString(ReadableMap frame, String key) {
        if (frame.hasKey(key)
                && !frame.isNull(key)
                && frame.getType(key) == ReadableType.Number) {
            return String.valueOf(frame.getInt(key));
        } else {
            return "-1";
        }
    }

    private static String concatFrame(
            String methodName, String fileId, String lineNumber, String columnNumber) {
        StringBuilder stringBuilder = new StringBuilder();
        stringBuilder.append(methodName).append("@").append(fileId).append(lineNumber);
        if (!columnNumber.equals("-1")) {
            stringBuilder.append(":").append(columnNumber);
        }
        stringBuilder.append("\n");
        return stringBuilder.toString();
    }

    private static String parseFileId(ReadableMap frame) {
        if (frame.hasKey(KEYS.FILE)
                && !frame.isNull(KEYS.FILE)
                && frame.getType(KEYS.FILE) == ReadableType.String) {
            String file = frame.getString(KEYS.FILE);
            if (file != null) {
                final Matcher matcher = FILE_ID_PATTERN.matcher(file);
                if (matcher.find()) {
                    return matcher.group(1) + ":";
                }
            }
        }
        return "";
    }
}