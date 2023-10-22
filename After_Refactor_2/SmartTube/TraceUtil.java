package com.google.android.exoplayer2.util;

import android.annotation.TargetApi;
import com.google.android.exoplayer2.ExoPlayerLibraryInfo;

/**
 * Calls through to {@link android.os.Trace} methods on supported API levels.
 */
public final class TraceUtil {

  private TraceUtil() {}

  /**
   * Writes a trace message to indicate that a given section of code has begun.
   *
   * @see android.os.Trace#beginSection(String)
   * @param sectionName The name of the code section to appear in the trace. This may be at most 127
   *     Unicode code units long.
   */
  public static void beginSection(String sectionName) {
    if (isTraceEnabled() && isApiLevelSupported(18)) {
      beginSectionV18(sectionName);
    }
  }

  /**
   * Writes a trace message to indicate that a given section of code has ended.
   *
   * @see android.os.Trace#endSection()
   */
  public static void endSection() {
    if (isTraceEnabled() && isApiLevelSupported(18)) {
      endSectionV18();
    }
  }

  private static boolean isTraceEnabled() {
    return ExoPlayerLibraryInfo.TRACE_ENABLED;
  }

  private static boolean isApiLevelSupported(int apiLevel) {
    return Util.SDK_INT >= apiLevel;
  }

  @TargetApi(18)
  private static void beginSectionV18(String sectionName) {
    android.os.Trace.beginSection(sectionName);
  }

  @TargetApi(18)
  private static void endSectionV18() {
    android.os.Trace.endSection();
  }

} 