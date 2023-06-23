package com.google.android.exoplayer2.util;

import android.annotation.TargetApi;
import android.os.Trace;

import com.google.android.exoplayer2.ExoPlayerLibraryInfo;

/**
 * Calls through to Trace methods on supported API levels.
 */
public final class TraceUtil {

  private static final int MINIMUM_SDK_VERSION_REQUIRED = 18;

  private TraceUtil() {}

  /**
   * Writes a trace message to indicate that a given section of code has begun.
   *
   * @param sectionName The name of the code section to appear in the trace. This may be at most 127
   *     Unicode code units long.
   */
  public static void beginSection(String sectionName) {
    if (isTraceEnabled() && isSupportedSdkVersion()) {
      beginTraceSection(sectionName);
    }
  }

  /**
   * Writes a trace message to indicate that a given section of code has ended.
   */
  public static void endSection() {
    if (isTraceEnabled() && isSupportedSdkVersion()) {
      endTraceSection();
    }
  }

  @TargetApi(MINIMUM_SDK_VERSION_REQUIRED)
  private static void beginTraceSection(String sectionName) {
    Trace.beginSection(sectionName);
  }

  @TargetApi(MINIMUM_SDK_VERSION_REQUIRED)
  private static void endTraceSection() {
    Trace.endSection();
  }

  private static boolean isTraceEnabled() {
    return ExoPlayerLibraryInfo.TRACE_ENABLED;
  }

  private static boolean isSupportedSdkVersion() {
    return Util.SDK_INT >= MINIMUM_SDK_VERSION_REQUIRED;
  }
} 