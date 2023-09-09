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
  private static final Pattern FILE_ID_PATTERN = Pattern.compile("\\b((?:seg-\\d+(?:_\\d+)?|\\d+)\\.js)");

  public static String formatStack(String message, ReadableArray stack) {
    StringBuilder stringBuilder = new StringBuilder(message).append(", stack:\n");
    for (int i = 0; i < stack.size(); i++) {
      ReadableMap frame = stack.getMap(i);
      String methodName = frame.getString(METHOD_NAME_KEY);
      String fileId = parseStackFrameFileId(frame);
      int lineNumber = parseStackFrameLineNumber(frame);
      int columnNumber = parseStackFrameColumnNumber(frame);

      stringBuilder.append(methodName).append("@").append(fileId).append(lineNumber);
      if (columnNumber != -1) {
        stringBuilder.append(":").append(columnNumber);
      }
      stringBuilder.append("\n");
    }
    return stringBuilder.toString();
  }

  private static String parseStackFrameFileId(ReadableMap frame) {
    if (frame.hasKey(FILE_KEY)
        && !frame.isNull(FILE_KEY)
        && frame.getType(FILE_KEY) == ReadableType.String) {
      String file = frame.getString(FILE_KEY);
      if (file != null) {
        final Matcher matcher = FILE_ID_PATTERN.matcher(file);
        if (matcher.find()) {
          return matcher.group(1) + ":";
        }
      }
    }
    return "";
  }

  private static int parseStackFrameLineNumber(ReadableMap frame) {
    switch (frame.getType(LINE_NUMBER_KEY)) {
      case Number:
        return frame.getInt(LINE_NUMBER_KEY);
      default:
        return -1;
    }
  }

  private static int parseStackFrameColumnNumber(ReadableMap frame) {
    switch (frame.getType(COLUMN_KEY)) {
      case Number:
        return frame.getInt(COLUMN_KEY);
      default:
        return -1;
    }
  }
} 