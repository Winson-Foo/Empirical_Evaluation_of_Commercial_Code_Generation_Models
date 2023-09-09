package com.facebook.react.bridge;

import android.os.SystemClock;
import androidx.annotation.Nullable;
import com.facebook.proguard.annotations.DoNotStrip;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArraySet;

@DoNotStrip
public class ReactMarker {

  public interface MarkerListener {
    void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey);
  }

  public interface FabricMarkerListener {
    void logFabricMarker(
        ReactMarkerConstants name, @Nullable String tag, int instanceKey, long timestamp);
  }

  private static final Set<MarkerListener> sListeners = new CopyOnWriteArraySet<>();

  private static final Set<FabricMarkerListener> sFabricMarkerListeners =
      new CopyOnWriteArraySet<>();

  private static long sAppStartTime;

  @DoNotStrip
  public static void addListener(MarkerListener listener) {
    sListeners.add(listener);
  }

  @DoNotStrip
  public static void removeListener(MarkerListener listener) {
    sListeners.remove(listener);
  }

  @DoNotStrip
  public static void clearMarkerListeners() {
    sListeners.clear();
  }

  @DoNotStrip
  public static void addFabricMarkerListener(FabricMarkerListener listener) {
    sFabricMarkerListeners.add(listener);
  }

  @DoNotStrip
  public static void removeFabricMarkerListener(FabricMarkerListener listener) {
    sFabricMarkerListeners.remove(listener);
  }

  @DoNotStrip
  public static void clearFabricMarkerListeners() {
    sFabricMarkerListeners.clear();
  }

  @DoNotStrip
  public static void logMarker(String name) {
    logMarker(name, null);
  }

  @DoNotStrip
  public static void logMarker(String name, int instanceKey) {
    logMarker(name, null, instanceKey);
  }

  @DoNotStrip
  public static void logMarker(String name, @Nullable String tag) {
    logMarker(name, tag, 0);
  }

  @DoNotStrip
  public static void logMarker(String name, @Nullable String tag, int instanceKey) {
    ReactMarkerConstants constants = ReactMarkerConstants.valueOf(name);
    logMarker(constants, tag, instanceKey);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants constants) {
    logMarker(constants, null, 0);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants constants, int instanceKey) {
    logMarker(constants, null, instanceKey);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants constants, @Nullable String tag) {
    logMarker(constants, tag, 0);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants constants, @Nullable String tag, int instanceKey) {
    logFabricMarker(constants, tag, instanceKey);
    for (MarkerListener listener : sListeners) {
      listener.logMarker(constants, tag, instanceKey);
    }
  }

  @DoNotStrip
  public static void setAppStartTime(long appStartTime) {
    sAppStartTime = appStartTime;
  }

  @DoNotStrip
  public static double getAppStartTime() {
    return (double) sAppStartTime;
  }

  @DoNotStrip
  public static void logFabricMarker(
      ReactMarkerConstants constants, @Nullable String tag, int instanceKey, long timestamp) {
    for (FabricMarkerListener listener : sFabricMarkerListeners) {
      listener.logFabricMarker(constants, tag, instanceKey, timestamp);
    }
  }

  @DoNotStrip
  public static void logFabricMarker(
      ReactMarkerConstants constants, @Nullable String tag, int instanceKey) {
    logFabricMarker(constants, tag, instanceKey, SystemClock.uptimeMillis());
  }
}