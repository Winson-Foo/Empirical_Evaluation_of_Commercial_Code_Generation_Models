public class ReactMarkerListener {

  private final List<MarkerListener> markerListeners = new CopyOnWriteArrayList<>();
  private final List<FabricMarkerListener> fabricMarkerListeners = new CopyOnWriteArrayList<>();

  public void addListener(MarkerListener listener) {
    if (!markerListeners.contains(listener)) {
      markerListeners.add(listener);
    }
  }

  public void removeListener(MarkerListener listener) {
    markerListeners.remove(listener);
  }

  public void clearMarkerListeners() {
    markerListeners.clear();
  }

  public void addFabricListener(FabricMarkerListener listener) {
    if (!fabricMarkerListeners.contains(listener)) {
      fabricMarkerListeners.add(listener);
    }
  }

  public void removeFabricListener(FabricMarkerListener listener) {
    fabricMarkerListeners.remove(listener);
  }

  public void clearFabricMarkerListeners() {
    fabricMarkerListeners.clear();
  }

  public void logFabricMarker(
      ReactMarkerConstants name, @Nullable String tag, int instanceKey, long timestamp) {
    for (FabricMarkerListener listener : fabricMarkerListeners) {
      listener.logFabricMarker(name, tag, instanceKey, timestamp);
    }
  }

  public void logFabricMarker(
      ReactMarkerConstants name, @Nullable String tag, int instanceKey) {
    logFabricMarker(name, tag, instanceKey, SystemClock.uptimeMillis());
  }

  public void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey) {
    logFabricMarker(name, tag, instanceKey);
    for (MarkerListener listener : markerListeners) {
      listener.logMarker(name, tag, instanceKey);
    }
  }
}
```

2. Remove unnecessary comments, and rename sListeners and sFabricMarkerListeners to markerListeners and fabricMarkerListeners respectively to comply with Java naming convention.

```
package com.facebook.react.bridge;

import android.os.SystemClock;
import androidx.annotation.Nullable;
import com.facebook.proguard.annotations.DoNotStrip;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@DoNotStrip
public class ReactMarker {

  public interface MarkerListener {
    void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey);
  }

  public interface FabricMarkerListener {
    void logFabricMarker(
        ReactMarkerConstants name, @Nullable String tag, int instanceKey, long timestamp);
  }

  private static final ReactMarkerListener reactMarkerListener = new ReactMarkerListener();

  private static long appStartTime;

  @DoNotStrip
  public static void addListener(MarkerListener listener) {
    reactMarkerListener.addListener(listener);
  }

  @DoNotStrip
  public static void removeListener(MarkerListener listener) {
    reactMarkerListener.removeListener(listener);
  }

  @DoNotStrip
  public static void clearMarkerListeners() {
    reactMarkerListener.clearMarkerListeners();
  }

  @DoNotStrip
  public static void addFabricListener(FabricMarkerListener listener) {
    reactMarkerListener.addFabricListener(listener);
  }

  @DoNotStrip
  public static void removeFabricListener(FabricMarkerListener listener) {
    reactMarkerListener.removeFabricListener(listener);
  }

  @DoNotStrip
  public static void clearFabricMarkerListeners() {
    reactMarkerListener.clearFabricMarkerListeners();
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
    ReactMarkerConstants marker = ReactMarkerConstants.valueOf(name);
    logMarker(marker, tag, instanceKey);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants name) {
    logMarker(name, null, 0);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants name, int instanceKey) {
    logMarker(name, null, instanceKey);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants name, @Nullable String tag) {
    logMarker(name, tag, 0);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey) {
    reactMarkerListener.logMarker(name, tag, instanceKey);
  }

  @DoNotStrip
  public static void setAppStartTime(long appStartTime) {
    ReactMarker.appStartTime = appStartTime;
  }

  @DoNotStrip
  public static double getAppStartTime() {
    return (double) appStartTime;
  }
}
```

3. Remove the redundant semicolon after the MarkerListener interface definition.

```
public interface MarkerListener {
  void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey);
};
```

becomes

```
public interface MarkerListener {
  void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey);
}
``` 

4. Use the standard Android naming convention for private members with a lowercase first letter.

```
private static final ReactMarkerListener reactMarkerListener = new ReactMarkerListener();
```

becomes

```
private static final ReactMarkerListener sReactMarkerListener = new ReactMarkerListener();
```

Here is the refactored code after all these changes:

```
package com.facebook.react.bridge;

import android.os.SystemClock;
import androidx.annotation.Nullable;
import com.facebook.proguard.annotations.DoNotStrip;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

@DoNotStrip
public class ReactMarker {

  public interface MarkerListener {
    void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey);
  }

  public interface FabricMarkerListener {
    void logFabricMarker(
        ReactMarkerConstants name, @Nullable String tag, int instanceKey, long timestamp);
  }

  private static final ReactMarkerListener sReactMarkerListener = new ReactMarkerListener();

  private static long sAppStartTime;

  @DoNotStrip
  public static void addListener(MarkerListener listener) {
    sReactMarkerListener.addListener(listener);
  }

  @DoNotStrip
  public static void removeListener(MarkerListener listener) {
    sReactMarkerListener.removeListener(listener);
  }

  @DoNotStrip
  public static void clearMarkerListeners() {
    sReactMarkerListener.clearMarkerListeners();
  }

  @DoNotStrip
  public static void addFabricListener(FabricMarkerListener listener) {
    sReactMarkerListener.addFabricListener(listener);
  }

  @DoNotStrip
  public static void removeFabricListener(FabricMarkerListener listener) {
    sReactMarkerListener.removeFabricListener(listener);
  }

  @DoNotStrip
  public static void clearFabricMarkerListeners() {
    sReactMarkerListener.clearFabricMarkerListeners();
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
    ReactMarkerConstants marker = ReactMarkerConstants.valueOf(name);
    logMarker(marker, tag, instanceKey);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants name) {
    logMarker(name, null, 0);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants name, int instanceKey) {
    logMarker(name, null, instanceKey);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants name, @Nullable String tag) {
    logMarker(name, tag, 0);
  }

  @DoNotStrip
  public static void logMarker(ReactMarkerConstants name, @Nullable String tag, int instanceKey) {
    sReactMarkerListener.logMarker(name, tag, instanceKey);
  }

  @DoNotStrip
  public static void setAppStartTime(long appStartTime) {
    sAppStartTime = appStartTime;
  }

  @DoNotStrip
  public static double getAppStartTime() {
    return (double) sAppStartTime;
  }
} 